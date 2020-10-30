import torch

def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (rgb, labels) in loader:
        ff = torch.FloatTensor(rgb.size(0), 1024).zero_()
        for i in range(2):
            if i == 1:
                rgb = rgb.index_select(3, torch.arange(rgb.size(3) - 1, -1, -1).long())
            input_img = rgb.to('cuda')
            outputs = model.C(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features
