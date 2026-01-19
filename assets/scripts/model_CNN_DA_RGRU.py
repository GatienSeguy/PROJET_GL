import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionModule(nn.Module):
    """Module de convolution avec batch normalization et activation"""
    def __init__(self, in_channels, out_channels, kernel_size=6):
        super(ConvolutionModule, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        return x


class AdditiveAttention(nn.Module):
    """Mécanisme d'attention additif (première couche d'attention)"""
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        Q = self.Wq(x)  # Query
        K = self.Wk(x)  # Key
        V = self.Wv(x)  # Value
        
        # Calcul des scores d'attention
        A = torch.matmul(Q, K.transpose(1, 2))  # (batch, seq_len, seq_len)
        A = F.softmax(A / (K.size(-1) ** 0.5), dim=-1)
        
        # Application de l'attention
        T = torch.matmul(A, V)  # (batch, seq_len, features)
        return T


class DoubleLayerAttention(nn.Module):
    """Module d'attention à deux couches"""
    def __init__(self, hidden_size):
        super(DoubleLayerAttention, self).__init__()
        self.attention1 = AdditiveAttention(hidden_size)
        self.attention2 = AdditiveAttention(hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        # Première couche d'attention
        alpha = self.attention1(x)
        x_prime = x * alpha
        
        # Activation
        x_double_prime = self.activation(x_prime)
        
        # Deuxième couche d'attention
        beta = self.attention2(x_double_prime)
        
        return beta, x_prime


class ResidualBlock(nn.Module):
    """Bloc résiduel"""
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = out + residual  # Connexion résiduelle
        return out


class RGRU(nn.Module):
    """GRU avec blocs résiduels"""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(RGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Blocs résiduels
        self.residual_block1 = ResidualBlock(input_size)
        self.residual_block2 = ResidualBlock(input_size)
        
        # Branche de convolution 1x1
        self.conv_branch = nn.Conv1d(input_size, input_size, kernel_size=1)
        
        # Couches GRU
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Branche principale avec blocs résiduels
        res1 = self.residual_block1(x)
        res2 = self.residual_block2(res1)
        
        # Branche résiduelle avec convolution 1x1
        x_transpose = x.transpose(1, 2)  # (batch, features, seq_len)
        branch = self.conv_branch(x_transpose)
        branch = branch.transpose(1, 2)  # (batch, seq_len, features)
        
        # Combinaison des branches
        combined = res2 + branch
        
        # Passage dans les GRU
        out, _ = self.gru1(combined)
        out, _ = self.gru2(out)
        
        return out


class CNN_DA_RGRU(nn.Module):
    """
    Modèle complet CNN-DA-RGRU pour prédiction de séries temporelles multivariées
    """
    def __init__(self, num_features, conv_channels=64, hidden_size=128, 
                 output_size=1, pred_steps=1, kernel_size=6, pool_size=2):
        super(CNN_DA_RGRU, self).__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pred_steps = pred_steps
        
        # Module de convolution
        self.conv_module = ConvolutionModule(num_features, conv_channels, kernel_size)
        
        # Module d'attention à deux couches
        self.double_attention = DoubleLayerAttention(conv_channels)
        
        # Module RGRU (GRU résiduel)
        self.rgru = RGRU(conv_channels, hidden_size)
        
        # Max pooling
        self.max_pool = nn.MaxPool1d(pool_size)
        
        # Couches fully connected
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size * pred_steps)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, num_features)
        Returns:
            predictions: Tensor de shape (batch_size, pred_steps, output_size)
        """
        # Module de convolution
        conv_out = self.conv_module(x)
        
        # Module d'attention à deux couches
        attention_out, x_prime = self.double_attention(conv_out)
        
        # Multiplication élément par élément
        attended = attention_out * x_prime
        
        # Module RGRU
        rgru_out = self.rgru(attended)
        
        # Max pooling sur la dimension temporelle
        pooled = rgru_out.transpose(1, 2)  # (batch, hidden_size, seq_len)
        pooled = self.max_pool(pooled)
        pooled = pooled.transpose(1, 2)    # (batch, seq_len_pooled, hidden_size)
        
        # Prendre la dernière sortie temporelle
        last_output = pooled[:, -1, :]  # (batch, hidden_size)
        
        # Couches fully connected
        h1 = self.relu(self.fc1(last_output))
        h1 = self.dropout(h1)
        h2 = self.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        out = self.fc3(h2)  # (batch, output_size * pred_steps)
        
        # Reshape pour obtenir les prédictions multi-step
        predictions = out.view(-1, self.pred_steps, self.output_size)
        
        return predictions


def count_parameters(model):
    """Compte le nombre de paramètres entraînables du modèle"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test du modèle
    batch_size = 32
    seq_len = 15
    num_features = 17  # 16 variables exogènes + 1 cible
    pred_steps = 6
    
    model = CNN_DA_RGRU(
        num_features=num_features,
        conv_channels=64,
        hidden_size=128,
        output_size=1,
        pred_steps=pred_steps,
        kernel_size=6,
        pool_size=2
    )
    
    print(f"Nombre de paramètres: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, num_features)
    y_pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Expected output shape: (batch_size={batch_size}, pred_steps={pred_steps}, output_size=1)")