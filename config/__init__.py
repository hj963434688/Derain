import argparse


train_path_list = ['../dataset/derain/my/train/', '../dataset/derain/fu/']
test_path_list = ['../dataset/derain/test/rain12/', '../dataset/derain/test/new10/']
logs_list = ['../logs/model_roi/', '../logs/roi/', '../logs/roi_fu/']


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_tag', type=int, help='训练数据集选择', default=0)
    parser.add_argument('--logs_tag', type=int, help='模型选择', default=1)

    parser.add_argument('--batch_size', type=int, help='批量大小', default=128)
    parser.add_argument('--patch_size', type=int, help='输入裁剪尺寸', default=33)
    parser.add_argument('--learning_rate', type=float, help='学习率', default=0.01)
    parser.add_argument('--lr_decay', type=float, help='学习率衰减', default=0.9)
    parser.add_argument('--decay_step', type=int, help='衰减间隔', default=10000)
    parser.add_argument('--max_step', type=int, help='训练迭代次数', default=500000)
    parser.add_argument('--save_step', type=int, help='保存模型间隔', default=2000)
    parser.add_argument('--log_step', type=int, help='输出信息间隔', default=50)
    parser.add_argument('--if_resort', type=bool, help='是否继续训练', default=False)

    return parser.parse_args(argv)


def parse_arguments_test(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_tag', type=int, help='测试数据集选择', default=0)
    parser.add_argument('--logs_tag', type=int, help='模型选择', default=0)

    parser.add_argument('--img_name', type=str, help='测试图片名称,无表示测试全部图', default=None)
    parser.add_argument('--save_img', type=str, help='是否保存结果图片', default=False)
    parser.add_argument('--save_data', type=str, help='是否保存结果数据', default=False)
    parser.add_argument('--result_path', type=str, help='保存路径', default='./result/')

    return parser.parse_args(argv)
