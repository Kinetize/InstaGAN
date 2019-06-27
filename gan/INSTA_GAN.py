import importlib
import torch
import torchvision
import PIL
import os
import matplotlib.pyplot as plt
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
    def __init__(self, noise_size, output_size, d=128):
        super(Generator, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(noise_size, d*8, 4, 1, 0)
        self.deconv1_bn = torch.nn.BatchNorm2d(d*8)
        self.deconv2 = torch.nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = torch.nn.BatchNorm2d(d*4)
        self.deconv3 = torch.nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = torch.nn.BatchNorm2d(d*2)
        self.deconv4 = torch.nn.ConvTranspose2d(d*2, 3, 4, 2, 1)
        
    def forward(self, input):
        input = input.expand(1, 1, -1, -1).permute(2, 3, 0, 1)
        x = torch.nn.functional.relu(self.deconv1_bn(self.deconv1(input)))
        x = torch.nn.functional.relu(self.deconv2_bn(self.deconv2(x)))
        x = torch.nn.functional.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x
 
class Discriminator(torch.nn.Module):
    def __init__(self, input_size=32, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = torch.nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = torch.nn.BatchNorm2d(d*2)
        self.conv3 = torch.nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = torch.nn.BatchNorm2d(d*4)
        self.conv4 = torch.nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.linear = torch.nn.Linear(1024 * 2 * 2, 1)

    def forward(self, input):
        x = torch.nn.functional.leaky_relu(self.conv1(input), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.linear(self.conv4(x).view(-1, 1024 * 2 * 2)))
        return x

def main(data_set, hyperparameters, device):
    image_path = os.path.join("..", "data")



    # hyperparameters
    img_size = hyperparameters["img_size"]
    noise_size = hyperparameters["noise_size"]
    lr = hyperparameters["lr"]
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]
    shuffle = hyperparameters["shuffle"]
    img_shape = hyperparameters["img_shape"]

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    img_flat_size = 3*img_size**2
    G = Generator(noise_size, img_flat_size).to(device)
    D = Discriminator(img_flat_size).to(device)

    BCE_loss = torch.nn.BCELoss()

    # Adam optimizer
    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)


    plt.ion()
    plt.figure()
    plt.show()
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        G_result = None
        for i, x_ in enumerate(data_loader):
            D.zero_grad()

    #        x_ = x_.view(-1, img_flat_size)
            
            current_batch_size = len(x_) 
            y_real_ = torch.ones(current_batch_size)
            y_fake_ = torch.zeros(current_batch_size)
            x_, y_real_, y_fake_ = torch.autograd.Variable(x_.to(device)), torch.autograd.Variable(y_real_.to(device)), torch.autograd.Variable(y_fake_.to(device))

            D_result = D(x_)
            D_real_loss = BCE_loss(D_result, y_real_)
            
            z_ = torch.randn((current_batch_size, noise_size))
            z_ = torch.autograd.Variable(z_.to(device))
            G_result = G(z_)

            D_result = D(G_result)
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            G.zero_grad()

            z_ = torch.randn((current_batch_size, noise_size))
            z_ = torch.autograd.Variable(z_.to(device))

            y_ = torch.ones(current_batch_size)
            y_ = torch.autograd.Variable(y_.to(device))

            G_result = G(z_)
            D_result = D(G_result)
            G_train_loss = BCE_loss(D_result, y_)
            G_train_loss.backward()
            G_optimizer.step()
            print("G_loss: " + str(G_train_loss) + " D_loss: " + str(D_train_loss))
        
        plt.imshow(G_result[0].cpu().permute(1,2,0).detach())
        plt.pause(0.01)





img_size = 32
img_shape = (img_size, img_size)
hyperparameters = {
    "img_size": img_size,
    "img_shape": img_shape,
    "noise_size": 100,
    "lr": 0.00001,
    "epochs": 100,
    "batch_size": 8,
    "shuffle": True
}


def load_data_set():
    print("Loading Images...")
    image_path = os.path.join("..", "data")
    data_set = InstaDataset(image_path, img_shape, num_img=None)
    return data_set


def show_data_set(data_set):
    plt.ion()
    plt.figure()
    plt.show()
    for img in data_set:
        plt.imshow(img)
        enter()

def reload(module):
    importlib.invalidate_caches()
    importlib.reload(module)

if __name__ == "__main__":
    main(data_set, hyperparameters, device)
