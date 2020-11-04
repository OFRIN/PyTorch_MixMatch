import torch
import torch.nn.functional as F

import numpy as np

class MixMatch:
    def __init__(self, model, T, alpha, rampup_length, batch_size, classes, iteration=1):
        self.model = model

        self.T = T
        self.alpha = alpha
        self.rampup_length = rampup_length

        self.iteration = iteration

        self.batch_size = batch_size
        self.classes = classes

    def get_linear_rampup(self):
        if self.rampup_length == 0:
            return 1.0
        else:
            return float(np.clip(self.iteration / self.rampup_length, 0.0, 1.0))

    def sharpening(self, logits):
        logits = logits**(1/self.T)
        return logits / logits.sum(dim=1, keepdims=True)

    def __call__(self, labeled_images, labels, unlabeled_images_list):
        labels = torch.zeros(self.batch_size, self.classes).scatter_(1, labels.view(-1,1).long(), 1)

        # 2~9
        with torch.no_grad():
            unlabeled_logits_list = [torch.softmax(self.model(unlabeled_images), dim=1) for unlabeled_images in unlabeled_images_list]
            unlabeled_logits = torch.mean(torch.FloatTensor(unlabeled_logits_list), dim=0)
            print(unlabeled_logits[0], unlabeled_logits.size()) # (64, 10)

            unlabeled_logits = self.sharpening(unlabeled_logits)
            print(unlabeled_logits[0], unlabeled_logits.size()) # (64, 10)

            unlabeled_logits = unlabeled_logits.detach()

        # 10,11
        all_images = torch.cat([labeled_images] + unlabeled_images_list, dim=0)
        all_labels = torch.cat([labels, unlabeled_logits, unlabeled_logits], dim=0)

        # 12
        indices = torch.randperm(all_images.size(0))

        images_a, images_b = all_images, all_images[indices]
        labels_a, labels_b = all_labels, all_labels[indices]

        # 13,14 - MixUp
        l = np.random.beta(self.alpha, self.alpha)
        l = max(l, 1-l)

        mixed_images = l * images_a + (1 - l) * images_b
        mixed_labels = l * labels_a + (1 - l) * labels_b

        # interleave for batch normalization
        mixed_images = list(torch.split(mixed_images, self.batch_size))
        mixed_images = self.interleave(mixed_images, self.batch_size)
        
        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # de-interleave for batch normalization
        logits = self.interleave(logits, batch_size)

        labeled_logits = logits[0]
        unlabeled_logits = torch.cat(logits[1:], dim=0)

        

    def interleave_offsets(batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets
    
    def interleave(xy, batch):
        nu = len(xy) - 1
        offsets = interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]