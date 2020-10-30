import random
import collections
from torch.utils.data import sampler
import torch

class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)
        self.cam_index = collections.defaultdict()
        self._id2cam = collections.defaultdict(list)

        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            cam = data_source.camera(path)
            self._id2index[_id].append(idx)
            self.cam_index[idx] = cam

        for key, value in self._id2index.items():
            for i in range(len(value)):
                self._id2cam[key].append(self.cam_index[value[i]])



    def __iter__(self):
        unique_ids = self.data_source.unique_ids

        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self._id2cam[_id], _id, self.batch_image))

        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, camera, id, k):

        if len(population) < k:
            population = population * k

        '''
        sample = []
        cam_max = [0]

        for i in range(len(camera)):
            if camera[i] == 1:
                cam_max.append(i)
                break
        for i in range(len(camera)):
            if camera[i] == 2:
                cam_max.append(i)
                break
        cam_max.append(len(camera))

        if id == 220:
            cam_max = [cam_max[0], cam_max[1] // 2, cam_max[1], cam_max[2]]


        for i in range(k):

            # # selected_index = int(torch.randint(low=cam_max[i], high=cam_max[i+1], size=(1,)).item())
            selected_index = int(torch.randint(low=cam_max[i // 2], high=cam_max[i // 2 + 1], size=(1,)).item())
            sample.append(population[selected_index])

        return sample
        '''
        return random.sample(population, k)

        # sample = []
        # index =int(torch.randint(low=0, high=3, size=(1, )))
        # selected_index=int(torch.randint(low=cam_max[index], high=cam_max[index+1], size=(1, )))
        # sample.append(population[selected_index])
        # selected_index = int(torch.randint(low=cam_max[index], high=cam_max[index + 1], size=(1,)))
        # sample.append(population[selected_index])
        # return sample
