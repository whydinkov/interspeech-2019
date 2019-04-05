from misc.constants import Labels


def get_channels_bias_avg(channel_ids, label_probabilities):
    current_channel_id = None
    current_group = {
        Labels.LEASTBIASED: 0,
        Labels.LEFT: 0,
        Labels.RIGHT: 0
    }
    result = []
    for channel_id, label_probability in zip(channel_ids, label_probabilities):
        leastbiased, left, right = label_probability

        # empty current group, initiate with current probabilities
        if current_channel_id is None:
            current_channel_id = channel_id
            current_group[Labels.LEASTBIASED] = leastbiased
            current_group[Labels.LEFT] = left
            current_group[Labels.RIGHT] = right

        # group contains the same channel, update group probabilities,
        # by addition
        elif current_channel_id == channel_id:
            current_group[Labels.LEASTBIASED] += leastbiased
            current_group[Labels.LEFT] += left
            current_group[Labels.RIGHT] += right

        # group ended, there is a new channel, aggregate results
        # initate the new channel
        else:
            result.append(most_probable_label(current_group))
            current_channel_id = channel_id
            current_group[Labels.LEASTBIASED] = leastbiased
            current_group[Labels.LEFT] = left
            current_group[Labels.RIGHT] = right

    if any(current_group.values()):
        result.append(most_probable_label(current_group))

    return result


def get_channels_bias_max(channel_ids, labels):
    current_channel_id = None
    current_group = {
        Labels.LEASTBIASED: 0,
        Labels.LEFT: 0,
        Labels.RIGHT: 0
    }
    result = []
    for channel_id, label_probabilities in zip(channel_ids, labels):
        leastbiased, left, right = label_probabilities
        if current_channel_id is None:
            current_channel_id = channel_id

            current_group[Labels.LEASTBIASED] = leastbiased
            current_group[Labels.LEFT] = left
            current_group[Labels.RIGHT] = right

        elif current_channel_id == channel_id:
            if leastbiased > current_group[Labels.LEASTBIASED]:
                current_group[Labels.LEASTBIASED] = leastbiased

            if right > current_group[Labels.RIGHT]:
                current_group[Labels.RIGHT] = right

            if left > current_group[Labels.LEFT]:
                current_group[Labels.LEFT] = left

        else:
            result.append(most_probable_label(current_group))

            current_channel_id = channel_id

            current_group[Labels.LEASTBIASED] = 0
            current_group[Labels.LEFT] = 0
            current_group[Labels.RIGHT] = 0

            leastbiased, left, right = label_probabilities

            if leastbiased > current_group[Labels.LEASTBIASED]:
                current_group[Labels.LEASTBIASED] = leastbiased

            if right > current_group[Labels.RIGHT]:
                current_group[Labels.RIGHT] = right

            if left > current_group[Labels.LEFT]:
                current_group[Labels.LEFT] = left

    if any(current_group.values()):
        result.append(most_probable_label(current_group))

    return result


def get_labels_from_proba(probabilities):
    to_labels = []
    for current_probabilities in probabilities:
        leastbiased, left, right = current_probabilities
        current_probabilities_dict = {
            Labels.LEASTBIASED: leastbiased,
            Labels.LEFT: left,
            Labels.RIGHT: right
        }

        to_labels.append(most_probable_label(current_probabilities_dict))

    return to_labels


def sort_by_default_distribution(labels):
    if not any(labels):
        raise Exception('Labels requrired.')

    if len(labels) == 1:
        raise Exception('Single label should not be sorted.')

    if Labels.LEASTBIASED in labels:
        return Labels.LEASTBIASED
    if Labels.RIGHT in labels:
        return Labels.RIGHT
    if Labels.LEFT in labels:
        return Labels.LEFT
    else:
        raise Exception('Not recognized label.')


def most_probable_label(label_prob_dict):
    top_labels = []
    top_probability = 0

    for label, label_probability in label_prob_dict.items():
        if label_probability > top_probability:
            top_labels = [label]
            top_probability = label_probability
            continue
        if label_probability == top_probability:
            top_labels.append(label)

    if len(top_labels) == 1:
        return top_labels[0]

    return sort_by_default_distribution(top_labels)
