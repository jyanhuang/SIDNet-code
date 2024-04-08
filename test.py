import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from test_data import TestData as data
from model import SIDNet
import torchvision.utils as utils


def main(test_img, test_gt, test_epoch):
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_data_dir = test_img
    test_data_gt = test_gt
    test_batch_size = 1
    network_height = 3
    network_width = 6
    num_dense_layer = 2
    growth_rate = 16
    test_data = data(test_data_dir, test_data_gt)
    test_data_loader = DataLoader(test_data, batch_size=test_batch_size)

    def save_image(dehaze, image_name, category):
        batch_num = len(dehaze)
        for ind in range(batch_num):
            print(image_name[ind][:-3])
            utils.save_image(dehaze[ind], './{}/{}'.format(category, image_name[ind][:-3] + 'jpg'))  # png  # 4-11/{}

    G1 = SIDNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
    G1 = G1.to(device)
    G1 = nn.DataParallel(G1, device_ids=device_ids)
    G1.load_state_dict(torch.load('./models/dense2_epoch'+str(test_epoch)+'_NYU.tar'))
    print("load model successful!")

    G1.eval()
    net_time = 0.
    net_count = 0.
    for batch_id, test_data in enumerate(test_data_loader):
        with torch.no_grad():
            haze, gt, image_name = test_data
            haze = haze.to(device)
            gt = gt.to(device)
            start_time = time.time()
            dehaze, _ = G1(haze)
            end_time = time.time() - start_time
            net_time += end_time
            net_count += 1

    # --- Save image --- #
        save_image(dehaze, image_name, 'NYUResult')

    print("all image saved!")

test_data_dir = "./data/test/real/"
test_data_gt = "./data/test/real/"
test_epoch = 199

main(test_data_dir, test_data_gt, test_epoch)



