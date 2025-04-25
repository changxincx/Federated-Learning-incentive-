import json
import time
import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
import torch.optim as optim
from sklearn.cluster import KMeans
import data_utils
import neural_nets
import distributed_training_utils as dst
from distributed_training_utils import Client, Server
import experiment_manager as xpm
import default_hyperparameters as dhp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import copy
torch.set_printoptions(threshold=5000)  # 设置threshold为一个很大的
import csv




#------------------------------------------------------比较相似性
def cosine_similarity(dict1, dict2):
  # 初始化两个向量列表
  vec1 = []
  vec2 = []

  # 遍历字典中的所有键，确保dict1和dict2具有相同的键
  for key in dict1:
    if key in dict2:
      # 获取张量，并确保它们在CPU上
      tensor1 = dict1[key].detach()
      tensor2 = dict2[key].detach()
      if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
      if tensor2.is_cuda:
        tensor2 = tensor2.cpu()

      # 将张量展平后添加到列表
      vec1.append(tensor1.numpy().flatten())
      vec2.append(tensor2.numpy().flatten())

  # 合并所有层的向量成一个长向量
  vec1 = np.concatenate(vec1)
  vec2 = np.concatenate(vec2)

  # 计算点积
  dot_product = np.dot(vec1, vec2)
  # 计算向量的欧氏范数
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  # 防止除以零
  # 添加对范数为零的检查
  if norm_vec1 == 0 or norm_vec2 == 0 or np.isnan(norm_vec1) or np.isnan(norm_vec2):
    return 0
  else:
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    if np.isnan(cosine_similarity):
      return 0
    else:
      return cosine_similarity
##-----------------------------------------------------------------------曼哈顿距离
import numpy as np

def manhattan_distance(dict1, dict2):
    # 初始化两个向量列表
    vec1 = []
    vec2 = []

    # 遍历字典中的所有键，确保dict1和dict2具有相同的键
    for key in dict1:
        if key in dict2:
            # 获取张量，并确保它们在CPU上
            tensor1 = dict1[key].detach()
            tensor2 = dict2[key].detach()
            if tensor1.is_cuda:
                tensor1 = tensor1.cpu()
            if tensor2.is_cuda:
                tensor2 = tensor2.cpu()

            # 将张量展平后添加到列表
            vec1.append(tensor1.numpy().flatten())
            vec2.append(tensor2.numpy().flatten())

    # 合并所有层的向量成一个长向量
    vec1 = np.concatenate(vec1)
    vec2 = np.concatenate(vec2)

    # 计算曼哈顿距离
    distance = np.sum(np.abs(vec1 - vec2))
    # 防止除以零，并处理无效值
    if np.isnan(distance) or distance == 0:
      return 1  # 如果距离为零，则相似度是最大的
    else:
      # 将欧几里得距离转换为相似度评分，这里我们使用1 / (1 + distance)的形式
      # 这样距离越小，相似度越大
      manhattan_distance = 1 / (1 + distance)
      if np.isnan(manhattan_distance):
        return 0
      else:
        return manhattan_distance

#--------------------------------------------------------------------汉明距离
def hamming_distance(dict1, dict2):
  # 初始化两个向量列表
  vec1 = []
  vec2 = []
  # 遍历字典中的所有键，确保dict1和dict2具有相同的键
  for key in dict1:
    if key in dict2:
      # 获取张量，并确保它们在CPU上
      tensor1 = dict1[key].detach()
      tensor2 = dict2[key].detach()
      if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
      if tensor2.is_cuda:
        tensor2 = tensor2.cpu()
      # 将张量展平后添加到列表
      vec1.append(tensor1.numpy().flatten())
      vec2.append(tensor2.numpy().flatten())
  # 合并所有层的向量成一个长向量
  vec1 = np.concatenate(vec1)
  vec2 = np.concatenate(vec2)
  # 计算汉明距离
  hamming_distance = np.sum(vec1 != vec2)
  return hamming_distance
# --------------------------------------------------------------------jaccard相似性
def jaccard_similarity(dict1, dict2):
  # 初始化两个集合
  set1 = set()
  set2 = set()

  # 遍历字典中的所有键，确保dict1和dict2具有相同的键
  for key in dict1:
    if key in dict2:
      # 获取张量，并确保它们在CPU上
      tensor1 = dict1[key].detach()
      tensor2 = dict2[key].detach()
      if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
      if tensor2.is_cuda:
        tensor2 = tensor2.cpu()

      # 将张量转换为numpy数组并展平
      array1 = tensor1.numpy().flatten()
      array2 = tensor2.numpy().flatten()

      # 将非零元素的索引添加到集合中
      non_zero_indices1 = np.where(array1 != 0)[0]
      non_zero_indices2 = np.where(array2 != 0)[0]
      set1.update(non_zero_indices1)
      set2.update(non_zero_indices2)

  # 计算两个集合的交集和并集
  intersection = set1.intersection(set2)
  union = set1.union(set2)
  # 防止除以零
  if len(union) == 0:
    return 0
  else:
    # 计算Jaccard相似度
    return len(intersection) / len(union)
#-----------------------------------------------------------------------tanm相似度
def tanimoto_similarity(dict1, dict2):
  # 初始化两个向量列表
  vec1 = []
  vec2 = []

  # 遍历字典中的所有键，确保dict1和dict2具有相同的键
  for key in dict1:
    if key in dict2:
      # 获取张量，并确保它们在CPU上
      tensor1 = dict1[key].detach()
      tensor2 = dict2[key].detach()
      if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
      if tensor2.is_cuda:
        tensor2 = tensor2.cpu()

      # 将张量展平后添加到列表
      vec1.append(tensor1.numpy().flatten())
      vec2.append(tensor2.numpy().flatten())

  # 合并所有层的向量成一个长向量
  vec1 = np.concatenate(vec1)
  vec2 = np.concatenate(vec2)

  # 计算点积
  dot_product = np.dot(vec1, vec2)
  # 计算向量的欧氏范数平方
  norm_vec1 = np.dot(vec1, vec1)
  norm_vec2 = np.dot(vec2, vec2)

  # 计算Tanimoto相似度
  if norm_vec1 == 0 and norm_vec2 == 0:
    return 1  # 如果两个向量都是零向量，我们可以认为它们是完全相同的
  return dot_product / (norm_vec1 + norm_vec2 - dot_product)
#------------------------------------------------------------切比雪夫相似度
def chebyshev_similarity(dict1, dict2):
  # 初始化两个向量列表
  vec1 = []
  vec2 = []
  # 遍历字典中的所有键，确保dict1和dict2具有相同的键
  for key in dict1:
    if key in dict2:
      # 获取张量，并确保它们在CPU上
      tensor1 = dict1[key].detach()
      tensor2 = dict2[key].detach()
      if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
      if tensor2.is_cuda:
        tensor2 = tensor2.cpu()
      # 将张量展平后添加到列表
      vec1.append(tensor1.numpy().flatten())
      vec2.append(tensor2.numpy().flatten())
  # 合并所有层的向量成一个长向量
  vec1 = np.concatenate(vec1)
  vec2 = np.concatenate(vec2)
  # 计算向量之间的切比雪夫距离
  distance = np.max(np.abs(vec1 - vec2))
  # 防止除以零，并处理无效值
  if np.isnan(distance) or distance == 0:
    return 1  # 如果距离为零，则相似度是最大的
  else:
    # 将切比雪夫距离转换为相似度评分，这里我们使用1 / (1 + distance)的形式
    # 这样距离越小，相似度越大
    chebyshev_similarity = 1 / (1 + distance)
    if np.isnan(chebyshev_similarity):
      return 0
    else:
      return distance
# ------------------------------------------------------------欧几里得相似度
def euclidean_similarity(dict1, dict2):
  # 初始化两个向量列表
  vec1 = []
  vec2 = []

  # 遍历字典中的所有键，确保dict1和dict2具有相同的键
  for key in dict1:
    if key in dict2:
      # 获取张量，并确保它们在CPU上
      tensor1 = dict1[key].detach()
      tensor2 = dict2[key].detach()
      if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
      if tensor2.is_cuda:
        tensor2 = tensor2.cpu()

      # 将张量展平后添加到列表
      vec1.append(tensor1.numpy().flatten())
      vec2.append(tensor2.numpy().flatten())

  # 合并所有层的向量成一个长向量
  vec1 = np.concatenate(vec1)
  vec2 = np.concatenate(vec2)

  # 计算向量之间的欧几里得距离
  distance = np.linalg.norm(vec1 - vec2)

  # 防止除以零，并处理无效值
  if np.isnan(distance) or distance == 0:
    return 1  # 如果距离为零，则相似度是最大的
  else:
    # 将欧几里得距离转换为相似度评分，这里我们使用1 / (1 + distance)的形式
    # 这样距离越小，相似度越大
    euclidean_similarity = 1 / (1 + distance)
    if np.isnan(euclidean_similarity):
      return 0
    else:
      return euclidean_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)

print("Torch Version: ", torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parser.parse_args()

# Load the Hyperparameters of all Experiments to be performed and set up the Experiments
with open('federated_learning.json') as data_file:
  experiments_raw = json.load(data_file)[args.schedule]

hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
if args.reverse_order:
  hp_dicts = hp_dicts[::-1]
experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]


def calculate_average_participation(clients, start_id, end_id):
  total_participation = 0

  num_clients = 0

  for client in clients:

    if start_id <= client.id <= end_id:
      total_participation += client.participation_count

      num_clients += 1

  return total_participation / num_clients if num_clients > 0 else 0

#-------------------------------------------slect and pay

#-------------------------------------------Greedy choice
def select_and_pay2(auctionable_clients, budget):
  # 筛选并按单位声誉价格排序
  best_selection = []
  best_total_payment = 0
  # 遍历所有可能的客户端组合z`
  for m in range(len(auctionable_clients)-1):
    selected = []
    total_payment = 0
    # 当前组合的定价标准
    current_price = auctionable_clients[m+1].bid
    for i in range(m + 1):
      # 计算支付额并添加到总支付
      payment = current_price
      total_payment += payment
      # 更新客户端的支付信息
      auctionable_clients[i].payment = payment
      selected.append(auctionable_clients[i])
    if total_payment > budget:
        break  # 超过预算，直接跳出循环
    # 如果总支付不超过预算并且比之前的总支付高，则更新最佳选择
    if total_payment <= budget and total_payment > best_total_payment:
      best_selection = selected
      best_total_payment = total_payment
      for client in best_selection:
         client.payment1 = current_price
  # 返回总支付和获胜者列表
  return best_total_payment, [c.id for c in best_selection]
def run_experiments(experiments):
  print("Running {} Experiments..\n".format(len(experiments)))
  output_file_path = "tensor.txt"  # 定义输出文件名
  for xp_count, xp in enumerate(experiments):
    hp = dhp.get_hp(xp.hyperparameters)
    xp.prepare(hp)
    print(xp)
    client_stats = {}
    average_par_round = 0
    # Load the Data and split it among the Clients
    client_loaders, train_loader, test_loader, stats = data_utils.get_data_loaders(hp)

    # Instantiate Clients and Server with Neural Net
    net = getattr(neural_nets, hp['net'])
    clients = [Client(loader, net().to(device), hp, xp, id_num=i) for i, loader in enumerate(client_loaders)]
    server = Server(test_loader, net().to(device), hp, xp, stats)

    # Print optimizer specs
    print_model(device=clients[0])
    print_optimizer(device=clients[0])

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    t1 = time.time()
    client_similarities = {}
    client_compression = {}
    group_similarities = {i: [] for i in range(400, 1000, 20)}
    exceedance_count = {client.id: 0 for client in clients}  # 存储每个客户端超出阈值的次数
    Sum_Round = 0
    Reputataion_Round = 0
    for c_round in range(1, hp['communication_rounds']+1):


      if c_round <= 350:
        # 前100轮从ID为0到150的客户端中选择
        eligible_clients = [client for client in clients if 0 <= client.id <= 600 or (800 <= client.id <= 900)]
        participating_clients = random.sample(eligible_clients, int(len(eligible_clients) * hp['participation_rate']))

      else:
        eligible_clients = [client for client in clients if 400 <= client.id <= 799 or (900 <= client.id <= 1000)]
        participating_clients = random.sample(eligible_clients, int(len(eligible_clients) * hp['participation_rate2']))




      for client in participating_clients:

        client.synchronize_with_server(server)
        client.compute_weight_update(hp['local_iterations'])
        client.compress_weight_update_up(compression=hp['compression_up'], accumulate=hp['accumulation_up'], count_bits=hp["count_bits"])
        client.participation_count += 1
      # Server does
      server.aggregate_weight_updates(participating_clients, aggregation=hp['aggregation'])
      server.compress_weight_update_down(compression=hp['compression_down'], accumulate=hp['accumulation_down'], count_bits=hp["count_bits"])



      current_round_similarities = {i: [] for i in range(400, 1000, 20)}


      # #------------------------------------------------------欧式距离评估
      # #------------------------------------------------------欧式距离评估

      for client in participating_clients:
        client.similarity_compressed = euclidean_similarity(client.dW_compressed, server.dW)  # 压缩dw的相似度，这个是原形
        # 如果字典中没有该客户端，初始化一个空列表
        if client.id not in client_similarities:
          client_similarities[client.id] = []
        # 将相似度添加到对应客户端的列表中
        client_similarities[client.id].append(client.similarity_compressed)
          # 分组计算平均相似度
        for group_start in range(400, 1000, 20):
          group_end = group_start + 20
          if group_start <= client.id < group_end:
            current_round_similarities[group_start].append(client.similarity_compressed)


      # -------------------------------------------生成图片的
      for group_start, similarities in current_round_similarities.items():
        if similarities:  # 确保有数据参与平均值计算
          average_similarities = np.mean(similarities)
        else:
          average_similarities = 0  # 如果没有数据，默认为0
        group_similarities[group_start].append(average_similarities)




      # Evaluate
      if xp.is_log_round(c_round):
        print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, len(experiments)))
        print("Evaluate...")
        results_train = server.evaluate(max_samples=5000, loader=train_loader)
        results_test = server.evaluate(max_samples=10000)

        # Logging
        xp.log({'communication_round' : c_round, 'lr' : clients[0].optimizer.__dict__['param_groups'][0]['lr'],
          'epoch' : clients[0].epoch, 'iteration' : c_round*hp['local_iterations']})
        xp.log({'client{}_loss'.format(client.id) : client.train_loss for client in clients}, printout=False)

        xp.log({key+'_train' : value for key, value in results_train.items()})
        xp.log({key+'_test' : value for key, value in results_test.items()})
        xp.log({'bits_sent_up': sum(participating_clients[10].bits_sent), 'bits_sent_down': sum(server.bits_sent)},
               printout=True)
        #
        xp.log({'time' : time.time()-t1}, printout=False)
        #
        # # Save results to Disk
        if 'log_path' in hp and hp['log_path']:
          xp.save_to_disc(path=hp['log_path'])
        #
        # Timing
        total_time = time.time()-t1
        avrg_time_per_c_round = (total_time)/c_round
        e = int(avrg_time_per_c_round*(hp['communication_rounds']-c_round))
        print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
                  "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))






    # -------------------------------------------生成图表。
    for client_id, similarities in client_similarities.items():
      plt.figure()
      plt.plot(range(1, 1 + len(similarities)), similarities)  # 从第351轮开始绘制
      plt.title(f"Similarity Over Rounds for Client {client_id}")
      plt.xlabel("Communication Round")
      plt.ylabel("Similarity")
      plt.grid(True)
      # 保存图表为PNG文件
      plt.savefig(f'./client_similarities/client_{client_id}_similarity.png')
      plt.close()  # 关闭图表以释放内存
      # 创建或打开 client_sim_file 文件夹下的文件，文件名格式为：client_{client_id}_similarity.csv
      filename = f'client_sim_file/client_{client_id}_similarity.csv'
      with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Round', 'Similarity'])
        # 写入每轮的相似度
        for round_num, similarity in enumerate(similarities, start=1):
          writer.writerow([round_num, similarity])
    # 假设 client_similarities 是一个字典，其中键是客户端ID，值是每轮的相似度列表



    for group_start, similarities in group_similarities.items():
      plt.figure()
      plt.plot(range(1, 1 + len(similarities)), similarities)
      plt.title(f"Average Euclidean similarity of 20 randomly selected clients")
      plt.xlabel("Communication Round")
      plt.ylabel("Average Similarity")
      plt.grid(True)
      plt.savefig(f'./client_similarities/average_similarity_clients_{group_start}_to_{group_start + 19}.png')
      plt.close()

      filename = f'sim_file/average_similarity_clients_{group_start}_to_{group_start + 19}.csv'
      with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 如果文件为空，则先写入表头
        if group_start == 400:
          writer.writerow(['Round', 'Similarity'])
        # 写入每轮的平均相似度
        for round_num, similarity in enumerate(similarities, start=1):
          writer.writerow([round_num, similarity])





    # Delete objects to free up GPU memory
    del server; clients.clear()
    torch.cuda.empty_cache()


def print_optimizer(device):
  try:
    print("Optimizer:", device.hp['optimizer'])
    for key, value in device.optimizer.__dict__['defaults'].items():
      print(" -", key,":", value)

    hp = device.hp
    base_batchsize = hp['batch_size']
    if hp['fix_batchsize']:
      client_batchsize = base_batchsize//hp['n_clients']
    else:
      client_batchsize = base_batchsize
    total_batchsize = client_batchsize*hp['n_clients']
    print(" - batchsize (/ total): {} (/ {})".format(client_batchsize, total_batchsize))
    print()
  except:
    pass


def print_model(device):
  print("Model {}:".format(device.hp['net']))
  n = 0
  for key, value in device.model.named_parameters():
    print(' -', '{:30}'.format(key), list(value.shape))
    n += value.numel()
  print("Total number of Parameters: ", n)
  print()


if __name__ == "__main__":
  run_experiments(experiments)