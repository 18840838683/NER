import bilsm_crf_model
import process_data
import numpy as np

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
# predict_text = '樊大志同志1987年8月参加工作。先后在东北财经大学、北京国际信托投资公司、北京市境外融投资管理中心、北京市国有资产经营有限责任公司、北京证券有限责任公司、北京首都创业集团有限公司、华夏银行股份有限公司工作。'
(train_x, train_y), (test_x, test_y), (vocab1, chunk_tags1) = process_data.load_data()

model.load_weights('model/crf.h5')


# str, length = process_data.process_data(predict_text, vocab)
# raw = model.predict(str)[0][-length:]
# print(raw)
pre_l = model.predict(test_x)
raw = [[np.argmax(row) for row in l]for l in pre_l]

tpre_nump,pre_nump=0,0
tpre_numl,pre_numl=0,0
tpre_numo,pre_numo=0,0
for l,r in zip(test_y,raw):
    for s,i in zip(l,r):
        if s == 1and i==1:
            tpre_nump += 1
        elif s == 3 and i==3:
            tpre_numl += 1
        elif s == 5and i==5:
            tpre_numo += 1

for l in raw:
    for s in l:
        if s == 1:
            pre_nump += 1
        elif s == 3:
            pre_numl += 1
        elif s == 5:
            pre_numo += 1

print("模型预测人名个数",pre_nump)
print("模型预测对的人名个数",tpre_nump)
print("模型预测地名个数",pre_numl)
print("模型预测对的地名个数",tpre_numl)
print("模型预测机构名个数",pre_numo)
print("模型预测对的机构名个数",tpre_numo)





# result = [np.argmax(row) for row in raw]
# result_tags = [chunk_tags[i] for i in result]
# print(result)
# per, loc, org = '', '', ''
#
# for s, t in zip(predict_text, result_tags):
#     if t in ('B-PER', 'I-PER'):
#         per += ' ' + s if (t == 'B-PER') else s
#     if t in ('B-ORG', 'I-ORG'):
#         org += ' ' + s if (t == 'B-ORG') else s
#     if t in ('B-LOC', 'I-LOC'):
#         loc += ' ' + s if (t == 'B-LOC') else s
#
# # print(['person:' + per, 'location:' + loc, 'organzation:' + org])
