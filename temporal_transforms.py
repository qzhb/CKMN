import random
import math


def center_get_frame_names(temp_frame_names, duration):
    center_index = len(temp_frame_names) // 2
    begin_index = max(0, center_index - (duration // 2))
    end_index = min(begin_index + duration, len(temp_frame_names))

    out = temp_frame_names[begin_index:end_index]

    for index in out:
        if len(out) >= duration:
            break
        out.append(out[-1])

    return out

def random_get_frame_names(temp_frame_names, duration):
    #import ipdb; ipdb.set_trace()
    rand_end = max(0, len(temp_frame_names) - duration - 1)
    begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + duration, len(temp_frame_names))

    out = temp_frame_names[begin_index:end_index]

    for index in out:
        if len(out) >= duration:
            break
        out.append(out[-1])

    return out


class TemporalSegmentCenterCrop(object):
    """Temporally segment the given frame indices and crop each segment at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, segment_number, sample_duration):
        self.segment_number = segment_number
        self.sample_duration = sample_duration

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = [frame_indice.strip() for frame_indice in frame_indices]

        if len(frame_indices) < self.segment_number:
            for i in range(self.segment_number - len(frame_indices)):
                frame_indices.append(frame_indices[-1])
            segment_frame_numbers = 1
        else:
            segment_frame_numbers = math.floor(len(frame_indices) / self.segment_number)

        frames_list = []
        for i in range(self.segment_number):
            temp_frames_list = []
            if i == self.segment_number - 1:
                temp_segment_frame_names = frame_indices[i * segment_frame_numbers:]
            else:
                temp_segment_frame_names = frame_indices[i * segment_frame_numbers: (i + 1) * segment_frame_numbers]

            segment_frame_names = center_get_frame_names(temp_segment_frame_names, self.sample_duration)

            for frame_name in segment_frame_names:
                temp_frames_list.append(frame_name)
            frames_list.append(temp_frames_list)

        return frames_list


class TemporalSegmentRandomCrop(object):
    """Temporally segment the given frame indices and crop each segment at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, segment_number, sample_duration):
        self.segment_number = segment_number
        self.sample_duration = sample_duration

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        #import ipdb; ipdb.set_trace()
        frame_indices = [frame_indice.strip() for frame_indice in frame_indices]
        
        if len(frame_indices) < self.segment_number:
            for i in range(self.segment_number - len(frame_indices)):
                frame_indices.append(frame_indices[-1])
            segment_frame_numbers = 1
        else:
            segment_frame_numbers = math.floor(len(frame_indices) / self.segment_number)

        frames_list = []
        for i in range(self.segment_number):
            temp_frames_list = []
            if i == self.segment_number - 1:
                temp_segment_frame_names = frame_indices[i * segment_frame_numbers:]
            else:
                temp_segment_frame_names = frame_indices[i * segment_frame_numbers: (i + 1) * segment_frame_numbers]

            segment_frame_names = random_get_frame_names(temp_segment_frame_names, self.sample_duration)

            for frame_name in segment_frame_names:
                temp_frames_list.append(frame_name)
            frames_list.append(temp_frames_list)
        #print(len(frames_list))
        return frames_list
