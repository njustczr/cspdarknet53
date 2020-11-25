import torch
from darknet53 import DarkNet53
import torch.nn.functional as F
import numpy as np
import time


if __name__ == '__main__':
    # distribute model ==> single model
    '''
    model_file = 'model_best.pth.tar'

    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

    model = DarkNet53(num_classes=1000)

    model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['state_dict'])

    torch.save(model.state_dict(), "darknet53.pth")
    '''
    #######################################

    with torch.no_grad():
        model = DarkNet53(num_classes=1000)
        model = torch.nn.DataParallel(model)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.load_state_dict(torch.load('darknet53.pth', map_location=device))
        model.eval()

        for i in range(10):
            start = time.time()
            input = torch.randn(1,3,256,256)
            predict = model(input)
            prob = F.softmax(predict)
            max_index = np.argmax(prob.cpu().numpy()[0])
            score = prob.cpu().numpy()[0][max_index]
            end = time.time()
            print(end-start)


