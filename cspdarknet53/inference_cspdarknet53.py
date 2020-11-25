import torch
from csdarknet53 import CsDarkNet53
import torch.nn.functional as F
import numpy as np
import time
import collections
import re

if __name__ == '__main__':
    with torch.no_grad():
        model = CsDarkNet53(num_classes=1000)
       
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.load_state_dict(torch.load('cspdarknet53.pth', map_location=device))
        
        model.eval()

        for i in range(10):
            start = time.time()
            input = torch.randn(1,3,256,256)
            predict = model(input)
            prob = F.softmax(predict)
            max_index = np.argmax(prob.cpu().numpy()[0])
            score = prob.cpu().numpy()[0][max_index]
            print(max_index, score)
            end = time.time()
            print(end-start)
