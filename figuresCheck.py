import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure()

columns = ["epoch", "loss", "val_loss"]
# columns = ["epoch", "loss"]
# df = pd.read_csv("/home/yuandou/test/CBIR/check/CAE.VAE.beforeTSNE/exp_0001/round_0001/prostate_cancer.csv", usecols=columns)
df = pd.read_csv("/home/yuandou/test/CBIR/check/Centralized.CAE.VAE.beforeTSNE/exp_0003/round_0001/prostate_cancer.csv", usecols=columns)
print("Contents in csv file: ", df)
# plt.plot(df.epoch, df.loss, df.val_loss)
plt.plot(df.epoch[1:24], df.loss[1:24], df.val_loss[1:24])
# plt.axis([0, df.epoch[0], 0, max(df.loss)])
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.show()

plt.savefig("/home/yuandou/test/CBIR/cbirtmp/train_check_0.png")

# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# # plt.axis([0, history.epoch[-1], 0, max(history.history['loss'] + history.history['val_loss'])])
# plt.axis([1, history.epoch[-1], 0, max(history.history['loss'] + history.history['val_loss'])])
# plt.legend(['loss', 'val_loss'], loc='upper right')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.savefig(run_folders["model_path"] + run_folders["exp_name"] + '/'+run_folders["round_name"]+"/viz/"+"training_loss.png")
# plt.savefig("/home/yuandou/test/CBIR/cbirtmp/train_check_1.png")
plt.close()
    