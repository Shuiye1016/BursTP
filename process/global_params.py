# 存储处理全局数据集的参数
def get_params(data_name='weibo'):
    if data_name == 'weibo':
        observation = 3600 * 1
        unit_time = 3600
        span = 24
        data_type = 'h'
        window_size = 60 * 30
        unit_size = 60

        rpath = '../dataset/weibo_part/dataset.txt'
        rppath = '../dataset/weibo_part/weibo_part.txt'
        wpath = '../dataset/weibo_part/'

    elif data_name == 'twitter':
        observation = 3600
        unit_time = 3600
        span = 24
        data_type = 'h'
        window_size = 60 * 30
        unit_size = 60

        rpath = '../dataset/twitter/dataset.txt'
        rppath = '../dataset/twitter/twitter.txt'
        wpath = '../dataset/twitter/'

    elif data_name == 'repost':
        observation = 3600
        unit_time = 3600
        span = 24
        data_type = 'h'
        window_size = 60 * 30
        unit_size = 60

        rpath = '../dataset/repost/dataset.txt'
        rppath = '../dataset/repost/repost.txt'
        wpath = '../dataset/repost/'

    elif data_name == 'topic':
        observation = 3600
        unit_time = 3600
        span = 24
        data_type = 'h'
        window_size = 60 * 30
        unit_size = 60

        rpath = '../dataset/topic/dataset.txt'
        rppath = '../dataset/topic/topic.txt'
        wpath = '../dataset/topic/'

    return observation, unit_time, span, window_size, unit_size, rpath, rppath, wpath, data_type