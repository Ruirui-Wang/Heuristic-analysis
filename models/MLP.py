import torch
import torch.nn as nn


class MLPPolynomial(nn.Module):
    def __init__(self, input_size, hidden_channels, num_series, A, nonlinearity=nn.ReLU, dropout=0.5):
        super(MLPPolynomial, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_series = num_series
        self.dropout = dropout
        # Define three separate linear layers
        self.fc1 = nn.Linear(input_size * num_series, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, 1)  # Final layer outputs a single value

        # Activation functions
        self.activation1 = nonlinearity()
        self.activation2 = nonlinearity()
        A = torch.tensor(A.toarray(), dtype=torch.float32)

        # Compute powers of adjacency matrix A
        self.A_powers = [A]  # A^1
        for n in range(1, num_series):
            self.A_powers.append(torch.matmul(self.A_powers[-1], A))  # A^n
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, i, j):
        assert torch.all((i >= 0) & (i < self.A_powers[0].shape[0])), "Index i is out of bounds."
        assert torch.all((j >= 0) & (j < self.A_powers[0].shape[0])), "Index j is out of bounds."

        # Perform element-wise multiplication without summing
        device = next(self.parameters()).device
        self.A_powers = [A_n.to(device) for A_n in self.A_powers]
        i, j = i.to(device), j.to(device)
        dot_products = [A_n[i] * A_n[j] for A_n in self.A_powers]

        # Concatenate along the series dimension
        A_combined = torch.cat(dot_products, dim=1)

        # Pass through the MLP
        output = self.fc1(A_combined)
        output = self.activation1(output)
        output = self.dropout_layer(output)

        output = self.fc2(output)
        output = self.activation2(output)
        output = self.dropout_layer(output)

        output = self.fc3(output)
        output = torch.sigmoid(output)
        return output.view(-1)



class MLPPolynomialFeatures(nn.Module):
    def __init__(self, input_size, feature_size, hidden_channels, num_series, A, nonlinearity=nn.ReLU, dropout=0.5):
        super(MLPPolynomialFeatures, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_series = num_series
        self.dropout = dropout
        # Define three separate linear layers
        self.fc1 = nn.Linear(input_size * num_series + feature_size, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, 1)  # Final layer outputs a single value

        # Activation functions
        self.activation1 = nonlinearity()
        self.activation2 = nonlinearity()
        A = torch.tensor(A.toarray(), dtype=torch.float32)

        # Compute powers of adjacency matrix A
        self.A_powers = [A]  # A^1
        for n in range(1, num_series):
            self.A_powers.append(torch.matmul(self.A_powers[-1], A))  # A^n
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, i, j, features_i, features_j):
        assert torch.all((i >= 0) & (i < self.A_powers[0].shape[0])), "Index i is out of bounds."
        assert torch.all((j >= 0) & (j < self.A_powers[0].shape[0])), "Index j is out of bounds."

        # Perform element-wise multiplication for each power of A
        device = next(self.parameters()).device
        self.A_powers = [A_n.to(device) for A_n in self.A_powers]
        i, j = i.to(device), j.to(device)
        dot_products_A = [A_n[i] * A_n[j] for A_n in self.A_powers]
        A_combined = torch.cat(dot_products_A, dim=1)  # Concatenate along series dimension

        # Perform element-wise multiplication for node features
        features_combined = features_i * features_j

        # Concatenate all inputs (A powers and node features)
        combined_input = torch.cat((A_combined, features_combined), dim=1)

        # Pass through the MLP
        output = self.fc1(combined_input)
        output = self.activation1(output)
        output = self.dropout_layer(output)

        output = self.fc2(output)
        output = self.activation2(output)
        output = self.dropout_layer(output)

        output = self.fc3(output)
        output = torch.sigmoid(output)
        return output.view(-1)





class MLPPolynomialLP(nn.Module):
    def __init__(self, input_size, hidden_channels, num_series, A, nonlinearity=nn.ReLU, dropout=0.5):
        super(MLPPolynomialLP, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_series = num_series
        self.dropout = dropout
        # Define three separate linear layers
        self.fc1 = nn.Linear(input_size * num_series, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, 1)  # Final layer outputs a single value

        # Activation functions
        self.activation1 = nonlinearity()
        self.activation2 = nonlinearity()
        A = torch.tensor(A.toarray(), dtype=torch.float32)

        # Compute powers of adjacency matrix A
        self.A_powers = [A]  # A^1
        for n in range(1, num_series):
            self.A_powers.append(torch.matmul(self.A_powers[-1], A))  # A^n
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, i, j):
        assert torch.all((i >= 0) & (i < self.A_powers[0].shape[0])), "Index i is out of bounds."
        assert torch.all((j >= 0) & (j < self.A_powers[0].shape[0])), "Index j is out of bounds."

        # Perform element-wise multiplication for each power of A
        dot_products = [A_n[i] * A_n[j] for A_n in self.A_powers]
        A_combined = torch.cat(dot_products, dim=1)  # Concatenate along series dimension

        # Pass through the MLP
        output = self.fc1(A_combined)
        output = self.activation1(output)
        output = self.dropout_layer(output)

        output = self.fc2(output)
        output = self.activation2(output)
        output = self.dropout_layer(output)

        output = self.fc3(output)
        output = torch.sigmoid(output)
        return output.view(-1)
