import torch
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from vector_wise_sparsity import create_mask

from apex.contrib.sparsity import ASP

from timeit import default_timer as timer

def main():

    torch.manual_seed(42)

    save_path = 'tmp/model/resnet50_cifar100'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = 128
    learning_rate = 0.1
    num_epochs = 350

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.vgg16(num_classes=100).to(device)
    # model = torchvision.models.resnet50(num_classes=100).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate

    ASP.prune_trained_model(model, optimizer)

    for epoch in range(num_epochs):


        for i, (images, labels) in enumerate(train_loader):
            
            start = timer()




            model.train()
            images = images.to(device)
            labels = labels.to(device)


            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            end = timer()

            print(end-start)
            if (i==16):
                return

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        # if (epoch+1) % 20 == 0:
        #     curr_lr /= 3
        #     update_lr(optimizer, curr_lr)
        if epoch==150:
            update_lr(optimizer,0.01)
        elif epoch == 250:
            update_lr(optimizer,0.001)

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        

    # Save the model checkpoint
        torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}.ckpt')

def main_transformer():
    import transformers

    device = 'cuda'

    config = transformers.BertConfig.from_pretrained('bert-base-uncased')
    model = transformers.BertForMaskedLM(config).to(device)

    rand_input = torch.randint(0,100,(8,512)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # ASP.prune_trained_model(model, optimizer)


    for i in range(16):

        start = timer()

        model_output = model(rand_input)

        optimizer.zero_grad()
        model_output[0].backward(torch.rand_like(model_output[0]))

        optimizer.step()

        end = timer()

        print(end-start)

if __name__ == "__main__":
    main()
    # main_transformer() 