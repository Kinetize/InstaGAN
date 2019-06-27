import torch
import torchvision
import PIL
import os
import matplotlib.pyplot as plt


image_path = os.path.join("..", "data")

class InstaDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, target_size, num_img=None):
        self.images = []
        img_files = list(os.listdir(image_path))
        for image in img_files[:num_img] if num_img is not None else img_files:
            img = PIL.Image.open(os.path.join(image_path, image)).convert("RGB")
            target_img = img.resize(target_size, PIL.Image.BILINEAR)
            img_tensor = torchvision.transforms.functional.to_tensor(target_img)
            self.images.append(img_tensor)
    def __getitem__(self, index):
        return self.images[index]
    def __len__(self):
        return len(self.images)


class Generator(torch.nn.Module):
    def __init__(self, noise_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(noise_size, 256)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, 512)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = torch.nn.Linear(self.fc3.out_features, output_size)
        
    def forward(self, input):
        x = torch.nn.functional.leaky_relu(self.fc1(input), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.2)
        x = torch.nn.functional.tanh(self.fc4(x))
        return x
 
class Discriminator(torch.nn.Module):
    def __init__(self, input_size=32):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 1024)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, 512)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, 256)
        self.fc4 = torch.nn.Linear(self.fc3.out_features, 1)

    def forward(self, input):
        x = torch.nn.functional.leaky_relu(self.fc1(input), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.sigmoid(self.fc4(x))
        return x


# hyperparameters
img_size = 28
noise_size = 10
lr = 0.001
epochs = 100
batch_size = 32
shuffle = True


img_shape = (img_size, img_size)

print("Loading Images...")
data_set = InstaDataset(image_path, img_shape, num_img=10)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

img_flat_size = 3*img_size**2
G = Generator(noise_size, img_flat_size)
D = Discriminator(img_flat_size)

BCE_loss = torch.nn.BCELoss()

# Adam optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)


plt.ion()
plt.figure()
plt.show()
for epoch in range(epochs):
    G_result = None
    for i, x_ in enumerate(data_loader):

        x_ = x_.view(-1, img_flat_size)
        
        current_batch_size = len(x_) 
        y_real_ = torch.ones(current_batch_size)
        y_fake_ = torch.zeros(current_batch_size)
        x_, y_real_, y_fake_ = torch.autograd.Variable(x_), torch.autograd.Variable(y_real_), torch.autograd.Variable(y_fake_)

        D_result = D(x_)

        print(len(x_))
        D_real_loss = BCE_loss(D_result, y_real_)
        
        z_ = torch.randn((current_batch_size, noise_size))
        z_ = torch.autograd.Variable(z_)
        G_result = G(z_)

        D_result = D(G_result)
        D_fake_loss = BCE_loss(D_result, y_fake_)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        G.zero_grad()

        z_ = torch.randn((current_batch_size, noise_size))
        z_ = torch.autograd.Variable(z_)

        y_ = torch.ones(current_batch_size)
        y_ = torch.autograd.Variable(y_)

        G_result = G(z_)
        D_result = D(G_result)
        G_train_loss = BCE_loss(D_result, y_)
        G_train_loss.backward()
        G_optimizer.step()
        print("G_loss: " + str(G_train_loss) + " D_loss: " + str(D_train_loss))
    
    plt.imshow(G_result[0].view(3, img_size, img_size).permute(1,2,0).detach())
    plt.pause(0.01)









