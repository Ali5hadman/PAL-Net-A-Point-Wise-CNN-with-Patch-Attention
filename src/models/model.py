
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim, time_steps):
        """
        Args:
            feature_dim (int): Number of features/channels in the input.
            time_steps (int): Length of the sequence (after any pooling).
        """
        super(Attention, self).__init__()
        # Weight: (feature_dim, 1), Bias: (time_steps, 1)
        self.W = nn.Parameter(torch.Tensor(feature_dim, 1))
        self.b = nn.Parameter(torch.Tensor(time_steps, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Use Xavier (Glorot) normal initialization for weights and zeros for bias
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, time_steps, feature_dim)
        Returns:
            Tensor: Weighted sum over time steps with shape (batch, feature_dim)
        """
        # Compute e = tanh(x @ W + b) where b is broadcasted along the batch dimension.
        # x: (batch, time, features), W: (features, 1) → (batch, time, 1)
        e = torch.tanh(torch.matmul(x, self.W) + self.b.unsqueeze(0))
        # Compute softmax along the time dimension (dim=1)
        a = F.softmax(e, dim=1)
        # Weighted sum over time steps
        weighted_input = x * a
        output = weighted_input.sum(dim=1)
        return output
    


class AttentionTopK(nn.Module):
    def __init__(self, feature_dim, time_steps, top_k=100):
        super(AttentionTopK, self).__init__()
        self.W = nn.Parameter(torch.Tensor(feature_dim, 1))
        self.b = nn.Parameter(torch.Tensor(time_steps, 1))
        self.top_k = top_k
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, time_steps, feature_dim)
        Returns:
            top_k_features: (batch, top_k, feature_dim)
            top_k_weights: (batch, top_k, 1)  # optional
        """
        # Compute attention scores (batch, time_steps, 1)
        e = torch.tanh(torch.matmul(x, self.W) + self.b.unsqueeze(0))

        # Flatten scores to (batch, time_steps)
        e = e.squeeze(-1)

        # Get top-K attention scores and their indices
        top_k_values, top_k_indices = torch.topk(e, self.top_k, dim=1)

        # Gather the top-K features based on indices
        batch_indices = torch.arange(x.size(0)).unsqueeze(-1).to(x.device)
        top_k_features = x[batch_indices, top_k_indices]  # (batch, top_k, feature_dim)

        # Optional: get the corresponding normalized weights (still using softmax over top-K)
        top_k_weights = F.softmax(top_k_values, dim=1).unsqueeze(-1)  # (batch, top_k, 1)

        # Optionally you can also return a weighted version of the top-K features:
        # weighted_top_k_features = top_k_features * top_k_weights

        return top_k_features, top_k_weights




class PALNET(nn.Module):
    def __init__(self, input_shape, output_shape, seed=42):
        """
        Args:
            input_shape (tuple): Expected as (height, width, channels) e.g. (50, 500, 3).
            output_shape (tuple): The shape to which the final Dense output is reshaped.
            seed (int, optional): Seed for weight initialization.
        """
        super(PALNET, self).__init__()
        torch.manual_seed(seed)
        # Since PyTorch expects (batch, channels, height, width), input channels = input_shape[2]
        in_channels = input_shape[2]
        
        # First block: two 1x1 conv layers with 32 filters, then maxpool with kernel (1, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        # Attention layer for the first block
        self.att1 = Attention(feature_dim=32, time_steps=input_shape[0]*input_shape[1])
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        
        # Second block: two 1x1 conv layers with 64 filters, then maxpool with kernel (1, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.att2 = Attention(feature_dim=64, time_steps=input_shape[0]* (input_shape[1]// 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        # Third block: two 1x1 conv layers with 128 filters, then maxpool with kernel (1, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        self.att3 = Attention(feature_dim=128, time_steps=input_shape[0]* ((input_shape[1]// 5)//5))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        
        # So the flattened size is 128 * 50 * 5 = 32000
        flattened_dim = 128 * input_shape[0] * (input_shape[1] // (5 * 5 * 4))  # 50 * 5 = 250; 128*250=32000
        flattened_dim = flattened_dim +32+64+128

        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 1024)  # linear activation
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, output_shape[0] * output_shape[1])
        self.output_shape = output_shape
        
        # Initialize weights with Xavier normal (Glorot normal)
        self._initialize_weights(seed)

    def _initialize_weights(self, seed=42):
        # If a seed is provided, set it for reproducibility in weight initialization.
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Expecting x shape: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x_att1 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att1 = self.att1(x_att1.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool1(x)
        

        x = self.conv2(x)
        x_att2 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att2 = self.att2(x_att2.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool2(x)


        x = self.conv3(x)
        x_att3 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att3 = self.att3(x_att3.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool3(x)

        x_flat = self.flatten(x)
        x = torch.cat([x_att1, x_att2, x_att3, x_flat], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.dropout(x)
        x = self.fc3(x)
        # Reshape output to desired output shape (excluding batch dimension)
        x = x.view(x.size(0), *self.output_shape)
        return x
    


#=================================================================================================================
# PLNET_noatt is the same as PLNET but without attention layers
#=================================================================================================================

class PLNET_noatt(nn.Module):
    def __init__(self, input_shape, output_shape, seed=42):
        """
        Args:
            input_shape (tuple): Expected as (height, width, channels) e.g. (50, 500, 3).
            output_shape (tuple): The shape to which the final Dense output is reshaped.
            seed (int, optional): Seed for weight initialization.
        """
        super(PLNET_noatt, self).__init__()
        torch.manual_seed(seed)
        # Since PyTorch expects (batch, channels, height, width), input channels = input_shape[2]
        in_channels = input_shape[2]
        
        # First block: two 1x1 conv layers with 32 filters, then maxpool with kernel (1, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        # Second block: two 1x1 conv layers with 64 filters, then maxpool with kernel (1, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        # Third block: two 1x1 conv layers with 128 filters, then maxpool with kernel (1, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        
        # So the flattened size is 128 * 50 * 5 = 32000
        flattened_dim = 128 * input_shape[0] * (input_shape[1] // (5 * 5 * 4))  # 50 * 5 = 250; 128*250=32000
        
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 1024)  # linear activation
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, output_shape[0] * output_shape[1])
        self.output_shape = output_shape
        
        # Initialize weights with Xavier normal (Glorot normal)
        self._initialize_weights(seed)

    def _initialize_weights(self, seed=42):
        # If a seed is provided, set it for reproducibility in weight initialization.
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Expecting x shape: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # Reshape output to desired output shape (excluding batch dimension)
        x = x.view(x.size(0), *self.output_shape)
        return x
    


#=================================================================================================================
# PALNET_topk is the same as PALNET but with top-k attention layers
#=================================================================================================================

class PALNET_topk(nn.Module):
    def __init__(self, input_shape, output_shape, seed=42):
        """
        Args:
            input_shape (tuple): Expected as (height, width, channels) e.g. (50, 500, 3).
            output_shape (tuple): The shape to which the final Dense output is reshaped.
            seed (int, optional): Seed for weight initialization.
        """
        super().__init__()
        torch.manual_seed(seed)
        H, W, C = input_shape
        self.top_k = 10
        
        # First block: two 1x1 conv layers with 32 filters, then maxpool with kernel (1, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        
        # one attention per row
        self.att1 = nn.ModuleList([
            AttentionTopK(feature_dim=32, time_steps=W, top_k=self.top_k)
            for _ in range(H)
        ])
        self.pool1 = nn.MaxPool2d(kernel_size=(1,5), stride=(1,5))
        

        # Second block: two 1x1 conv layers with 64 filters, then maxpool with kernel (1, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # one attention per row
        self.att2 = nn.ModuleList([
            AttentionTopK(feature_dim=64, time_steps= input_shape[1]// 5, top_k=self.top_k)
            for _ in range(H)
        ])
        
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        # Third block: two 1x1 conv layers with 128 filters, then maxpool with kernel (1, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        # self.att3 = AttentionTopK(feature_dim=128, time_steps=input_shape[0]* ((input_shape[1]// 5)//5), top_k=self.top_k)
        # one attention per row
        self.att3 = nn.ModuleList([
            AttentionTopK(feature_dim=128, time_steps= (input_shape[1]// 5)//5, top_k=self.top_k)
            for _ in range(H)
        ])
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        
        # So the flattened size is 128 * 50 * 5 = 32000
        flattened_dim = 128 * input_shape[0] * (input_shape[1] // (5 * 5 * 4))  # 50 * 5 = 250; 128*250=32000
        flattened_dim = flattened_dim +(self.top_k*32* H)+(self.top_k*64* H) +(self.top_k*128* H)

        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 1024)  # linear activation
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, output_shape[0] * output_shape[1])
        self.output_shape = output_shape
        
        # Initialize weights with Xavier normal (Glorot normal)
        self._initialize_weights(seed)

    def _initialize_weights(self, seed=42):
        # If a seed is provided, set it for reproducibility in weight initialization.
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Expecting x shape: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)

        
        # ─── conv1 + per-row attention ───────────────────────
        x1 = self.conv1(x)   # (batch,32,H,W)
        batch_size, _, H, W = x1.shape

        att1_feats = []
        for row_idx, attn in enumerate(self.att1):
            row_feats = x1[:,:,row_idx,:]                # (batch,32,W)
            row_feats = row_feats.permute(0,2,1)          # (batch,W,32)
            top_feats, _ = attn(row_feats)                # (batch,top_k,32)
            att1_feats.append(top_feats.reshape(batch_size, -1))
        x_att1 = torch.cat(att1_feats, dim=1)             # (batch, top_k*32*H)

        x = self.pool1(x1)                                # (batch,32,H, W//5)
        

        # ─── conv2 + per-row attention ───────────────────────
        x2 = self.conv2(x)   # (batch,32,H,W)
        batch_size, _, H, W = x1.shape

        att2_feats = []
        for row_idx, attn in enumerate(self.att2):
            row_feats = x2[:,:,row_idx,:]                # (batch,32,W)
            row_feats = row_feats.permute(0,2,1)          # (batch,W,32)
            top_feats, _ = attn(row_feats)                # (batch,top_k,32)
            att2_feats.append(top_feats.reshape(batch_size, -1))
        x_att2 = torch.cat(att2_feats, dim=1)             # (batch, top_k*32*H)

        x = self.pool2(x2)                                # (batch,32,H, W//5)


        # ─── conv3 + per-row attention ───────────────────────
        x3 = self.conv3(x)   # (batch,32,H,W)
        batch_size, _, H, W = x1.shape

        att3_feats = []
        for row_idx, attn in enumerate(self.att3):
            row_feats = x3[:,:,row_idx,:]                # (batch,32,W)
            row_feats = row_feats.permute(0,2,1)          # (batch,W,32)
            top_feats, _ = attn(row_feats)                # (batch,top_k,32)
            att3_feats.append(top_feats.reshape(batch_size, -1))
        x_att3 = torch.cat(att3_feats, dim=1)             # (batch, top_k*32*H)

        x = self.pool3(x3)                                # (batch,32,H, W//5)

        x_flat = self.flatten(x)
        x = torch.cat([x_att1, x_att2, x_att3, x_flat], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.dropout(x)
        x = self.fc3(x)
        # Reshape output to desired output shape (excluding batch dimension)
        x = x.view(x.size(0), *self.output_shape)
        return x
    

#=================================================================================================================
# PALNET_2blk is the same as PALNET but with only two blocks 
#=================================================================================================================

class PALNET_2blk(nn.Module):
    def __init__(self, input_shape, output_shape, seed=42):
        """
        Args:
            input_shape (tuple): Expected as (height, width, channels) e.g. (50, 500, 3).
            output_shape (tuple): The shape to which the final Dense output is reshaped.
            seed (int, optional): Seed for weight initialization.
        """
        super(PALNET_2blk, self).__init__()
        torch.manual_seed(seed)
        # Since PyTorch expects (batch, channels, height, width), input channels = input_shape[2]
        in_channels = input_shape[2]
        
        # First block: two 1x1 conv layers with 32 filters, then maxpool with kernel (1, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        # Attention layer for the first block
        self.att1 = Attention(feature_dim=32, time_steps=input_shape[0]*input_shape[1])
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        
        # Second block: two 1x1 conv layers with 64 filters, then maxpool with kernel (1, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.att2 = Attention(feature_dim=64, time_steps=input_shape[0]* (input_shape[1]// 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))

        
        # So the flattened size is 128 * 50 * 5 = 32000
        flattened_dim = input_shape[0] * (input_shape[1] // 25) * 64
        flattened_dim = flattened_dim +32+64

        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 1024)  # linear activation
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, output_shape[0] * output_shape[1])
        self.output_shape = output_shape
        
        # Initialize weights with Xavier normal (Glorot normal)
        self._initialize_weights(seed)

    def _initialize_weights(self, seed=42):
        # If a seed is provided, set it for reproducibility in weight initialization.
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Expecting x shape: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x_att1 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att1 = self.att1(x_att1.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool1(x)
        

        x = self.conv2(x)
        x_att2 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att2 = self.att2(x_att2.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool2(x)


        x_flat = self.flatten(x)
        x = torch.cat([x_att1, x_att2, x_flat], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.dropout(x)
        x = self.fc3(x)
        # Reshape output to desired output shape (excluding batch dimension)
        x = x.view(x.size(0), *self.output_shape)
        return x
    



#=================================================================================================================
# PALNET_ndo is the same as PALNET but with no dropout layers
#=================================================================================================================

class PALNET_ndo(nn.Module):
    def __init__(self, input_shape, output_shape, seed=42):
        """
        Args:
            input_shape (tuple): Expected as (height, width, channels) e.g. (50, 500, 3).
            output_shape (tuple): The shape to which the final Dense output is reshaped.
            seed (int, optional): Seed for weight initialization.
        """
        super(PALNET_ndo, self).__init__()
        torch.manual_seed(seed)
        # Since PyTorch expects (batch, channels, height, width), input channels = input_shape[2]
        in_channels = input_shape[2]
        
        # First block: two 1x1 conv layers with 32 filters, then maxpool with kernel (1, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        # Attention layer for the first block
        self.att1 = Attention(feature_dim=32, time_steps=input_shape[0]*input_shape[1])
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        
        # Second block: two 1x1 conv layers with 64 filters, then maxpool with kernel (1, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.att2 = Attention(feature_dim=64, time_steps=input_shape[0]* (input_shape[1]// 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        
        # Third block: two 1x1 conv layers with 128 filters, then maxpool with kernel (1, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        self.att3 = Attention(feature_dim=128, time_steps=input_shape[0]* ((input_shape[1]// 5)//5))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        
        # So the flattened size is 128 * 50 * 5 = 32000
        flattened_dim = 128 * input_shape[0] * (input_shape[1] // (5 * 5 * 4))  # 50 * 5 = 250; 128*250=32000
        flattened_dim = flattened_dim +32+64+128

        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 1024)  # linear activation
        self.fc3 = nn.Linear(1024, output_shape[0] * output_shape[1])
        self.output_shape = output_shape
        
        # Initialize weights with Xavier normal (Glorot normal)
        self._initialize_weights(seed)

    def _initialize_weights(self, seed=42):
        # If a seed is provided, set it for reproducibility in weight initialization.
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Expecting x shape: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x_att1 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att1 = self.att1(x_att1.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool1(x)
        

        x = self.conv2(x)
        x_att2 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att2 = self.att2(x_att2.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool2(x)


        x = self.conv3(x)
        x_att3 = x.view(x.shape[0], x.shape[1], x.shape[2]* x.shape[3])  # (batch, L1, 32)
        x_att3 = self.att3(x_att3.permute(0,2,1))  # (batch, L1, 32)
        x = self.pool3(x)

        x_flat = self.flatten(x)
        x = torch.cat([x_att1, x_att2, x_att3, x_flat], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.fc3(x)
        # Reshape output to desired output shape (excluding batch dimension)
        x = x.view(x.size(0), *self.output_shape)
        return x
    
