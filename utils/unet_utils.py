import torch
import torch.nn.functional as F
from monai.metrics import DiceHelper
from tqdm import tqdm

def test_unet(unet, testloader, organs, device):
    '''
    args:
        unet: The U-Net model.
        testloader: Data loader for testing.
        organs: List of organs.
        device: Computing device (e.g., 'cuda' or 'cpu').
    
    returns:
        dice_list: (list) Dice score for each organ for every sample in the testloader.
    '''

    unet.eval() # OFF bathnorm, dropout
    unet.to(device)

    dice_list = []
    dice_calculator = DiceHelper(
        include_background=True,
        get_not_nans=True,
    )
    
    with torch.no_grad(): # no backprop to economize memory
        for idx, d in tqdm(enumerate(testloader)):
            images, labels = d
            images, labels = images.unsqueeze(1).to(device), labels.to(device)
            outputs = unet(images)
            outputs = F.softmax(outputs, dim=1)

            l, o  = labels, outputs
            l = F.one_hot(l, num_classes=len(organs)).movedim(-1, 1)

            outputs = outputs.argmax(dim=1)
            
            for i in range(len(images)):
                dice_score, not_nans = dice_calculator(o[i:i+1], l[i:i+1])
                dice_score = dice_score.cpu().detach().numpy()
                dice_list.append(dice_score)

    return dice_list