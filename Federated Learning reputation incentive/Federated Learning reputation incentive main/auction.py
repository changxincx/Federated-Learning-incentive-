import random


# 简化版的Candidate类
class SimpleCandidate:
    def __init__(self, id, initial_bid):
        self.id = id
        self.bid = initial_bid
        self.reputation = []  # 这里我们将使用一个列表来存储声誉历史
        self.generate_random_reputation()
        self.calculate_unit_price()
        self.payment = None
        self.payment1 = None

    def generate_random_reputation(self):
        # 生成随机的声誉历史
        num_rounds = random.randint(1, 10)  # 随机确定声誉历史的长度
        self.reputation = [random.uniform(0.56, 0.95) for _ in range(num_rounds)]

    def calculate_unit_price(self):
        # 使用reputation列表的最后一个元素作为current_reputation
        if self.reputation:
            self.current_reputation = self.reputation[-1]
            if self.current_reputation != 0:
                self.unit_reputation_price = self.bid / self.current_reputation
            else:
                self.unit_reputation_price = float('inf')
        else:
            self.current_reputation = 0
            self.unit_reputation_price = 0


# 创建候选者列表
auctionable_clients = []
for i in range(1, 501):  # 创建5个候选人
    bid = random.uniform(3.0, 5.0)  # 生成随机出价
    auctionable_clients.append(SimpleCandidate(i, bid))

# 预算
budget = 300
# 输出所有候选者的信息

print("Initial Clients Information:")
auctionable_clients.sort(key=lambda c: c.unit_reputation_price)  # 排序
for client in auctionable_clients:
    print(f"Client ID: {client.id}, Bid: {client.bid:.2f}, reputataon:{client.current_reputation}, Unit Reputation Price: {client.unit_reputation_price:.2f}")

# 选择并支付候选人
def select_and_pay(auctionable_clients, budget):
  # 筛选并按单位声誉价格排序
  best_selection = []
  best_total_payment = 0
  # 遍历所有可能的客户端组合z`
  for m in range(len(auctionable_clients)-1):
    selected = []
    total_payment = 0
    # 当前组合的定价标准
    current_price = auctionable_clients[m+1].unit_reputation_price
    for i in range(m + 1):
      # 计算支付额并添加到总支付
      payment = auctionable_clients[i].current_reputation * current_price
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
      for i in range(m + 1):
        auctionable_clients[i].payment1 = auctionable_clients[i].payment
  # 返回总支付和获胜者列表
  return best_total_payment, [c.id for c in best_selection]

def select_and_pay2(auctionable_clients, budget):  # 价格贪婪的选择
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
      payment =  current_price
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
# 单位声誉价格执行选择和支付
# auctionable_clients.sort(key=lambda c: c.unit_reputation_price)
# best_total_payment, winners_ids  = select_and_pay(auctionable_clients, budget)

# 价格贪婪执行选择和支付
# auctionable_clients.sort(key=lambda c: c.bid)
# best_total_payment, winners_ids  = select_and_pay2(auctionable_clients, budget)

#价格随机执行选择和支付
best_total_payment, winners_ids  = select_and_pay(auctionable_clients, budget)


# 从auctionable_clients列表中获取获胜者的详细信息
winners_details = [
    {
        'id': client.id,
        'bid': client.bid,
        'payment': client.payment1,
        'unit_reputation_price': client.unit_reputation_price
    }
    for client in auctionable_clients if client.id in winners_ids
]
# 输出结果
print("Payment Record:")
print(f"Total Payment: {best_total_payment}")

print("Winners Details:")
for detail in winners_details:
    print(detail)
print(f"Number of Winners: {len(winners_ids)}")  # 添加这一行来输出获胜者人数
