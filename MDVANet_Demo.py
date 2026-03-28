import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding
from layers.Embed import PeriodicAwareConv
from layers.Embed import Dilated3DConvForTS
from layers.Embed import Advanced3DConvForTS

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.is_train = configs.is_training
        self.use_mask = True
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # Add learnable fusion weight parameters.
        self.param_1d = nn.Parameter(torch.rand(1))  # [0,1] random
        self.param_2d = nn.Parameter(torch.rand(1))

        # self.param_1d = nn.Parameter(torch.empty(1).uniform_(0.0, 0.5))  # 0～0.5

        # Initialize the periodicity-aware convolution.（2D）
        self.periodic_conv = PeriodicAwareConv(
            in_channels=1, 
            hidden_dim=configs.d_ff // 2,  # Use an appropriate hidden layer dimension.
            # hidden_dim =256,
            out_channels=1,
            intra_kernel=5,  # Intra-period convolution kernel size.
            inter_kernel=5   # Inter-period convolution kernel size.
        )

        # 3D
        self.conv3d = Advanced3DConvForTS(
            in_channels=1,
            hidden_dim=2,
            out_channels=1,
            kernel_size=(3,3,5),  
            dilation_rate=(1,2,2),
            dropout_rate=0.3
        )

 
    def forward(self, x):
        B, T, N = x.size()
        batch_size = x.shape[0]
        # normalization and permute 
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        # print(x.shape)  # Debug information, Output：[128, 7, 720] (B,N,T)

        # Period-level mask
        if self.is_train and self.use_mask:
            mask = torch.ones_like(x)
            # Calculate the total number of periods
            total_periods = T // self.period_len
            
            # The first 70% of periods
            front_periods = int(total_periods * 0.7)
            
            # Calculate the number of periods to mask（30%）
            mask_periods_count = int(front_periods * 0.3)

            # Optimization
            period_indices = torch.randperm(front_periods)[:mask_periods_count]
            # Apply the same masking pattern to all samples
            for period_idx in period_indices:
                # Start and end indices of a period
                start_idx = period_idx * self.period_len
                end_idx = (period_idx + 1) * self.period_len
                mask[:, :, start_idx:end_idx] = 0

            x_mask = x * mask
        
        # Apply mask.
        x_1d = x_mask.clone()
        x_2d = x_mask.clone()

        # x_1d = x.clone()
        # x_2d = x.clone()
        x_3d = x.clone()

        # 1D convolution aggregation
        x_1d = self.conv1d(x_1d.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x_1d
        x_1d = x_1d.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # 2D
        m_2d = x_2d.reshape(-1, T).reshape(-1, self.seg_num_x, self.period_len)  # BxN,gs,p_len
        # 2D conv
        m_2dconv = self.periodic_conv(m_2d)  # Apply periodicity-aware convolution.
        # Reshape the convolution output to the same shape as x for subsequent prediction.
        m_2dreshaped = m_2dconv.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # 3D
        window_size = 3  # Number of large windows.
        sub_window =5   # Number of small windows per large window.
        segment_len =48  # Length of each small window.

        # First, reshape the sequence into [B*N, window_size*sub_window, segment_len]
        temp = x_3d.reshape(-1, self.seq_len)
        # Reshape into a 3D temporal structure: [B*N, window_size, sub_window*segment_len]
        temp = temp.reshape(-1, window_size, sub_window * segment_len)
        
        # Further reshape: [B*N, window_size, sub_window, segment_len]
        temp = temp.reshape(-1, window_size, sub_window, segment_len)
        
        # Add a channel dimension: [B*N, 1, window_size, sub_window, segment_len]
        temp = temp.unsqueeze(1)

        n_3d = temp.clone()

        # Apply 3D convolution.
        n3d_conv = self.conv3d(temp) 

        #  Flatten the temporal dimension.
        n_flat = n3d_conv.squeeze(1).reshape(-1, window_size * sub_window * segment_len)
        # Reshape
        m_3dreshaped = n_flat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        y_1d = self.mlp(x_1d)
        y_2d = self.mlp(m_2dreshaped)
        y_3d = self.mlp(m_3dreshaped)

        # upsampling
        y_1d = y_1d.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y_2d = y_2d.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y_3d = y_3d.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        
        y_1d = y_1d.permute(0, 2, 1) + seq_mean
        y_2d = y_2d.permute(0, 2, 1) + seq_mean
        y_3d = y_3d.permute(0, 2, 1) + seq_mean

        # Use Sigmoid to constrain the learnable weight alpha between 0 and 1.
        alpha_1d = torch.sigmoid(self.param_1d)
        param_2d = torch.sigmoid(self.param_2d)
        
        # Ensure the weights sum to 1.
        total = alpha_1d + param_2d + (1 - alpha_1d - param_2d)
        w1 = alpha_1d / total          # 1D
        w2 = param_2d / total           # 2D
        w3 = (1 - alpha_1d - param_2d) / total  # 3D
        
        result = w1 * y_1d + w2 * y_2d + w3 * y_3d

        return result
