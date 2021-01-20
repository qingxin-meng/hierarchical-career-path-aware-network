import time
from module.utils import *
from ipdb import set_trace


class evaluate():
    def __init__(self,validate_last=False):
        self.max_time = 21
        self.total_time_hamming = 0.0
        self.total_time_pow = 0.0
        self.total_predict_num = 0.0
        self.correct_num_top1 = 0.0
        self.correct_num_top15 = 0.0
        self.correct_num_top30 = 0.0
        self.mrr_sum = 0.0
        self.acc_top1 = None
        self.acc_top15 = None
        self.acc_top30 = None
        self.time_mae = None
        self.time_rmse = None
        self.validate_last=validate_last

    def batch_evaluate(self, delta_lambda, prob_next_event, mask, target_event, target_time):
        if self.validate_last is False:
            predict_num = np.sum(mask)
            for prob_one_seq, mask_one_seq, target_one_seq in zip(prob_next_event, mask, target_event):
                for prob_one_event, target in zip(prob_one_seq[np.where(mask_one_seq == 1)],
                                                  target_one_seq[np.where(mask_one_seq == 1)]):
                    prob_sort = np.argsort(prob_one_event)
                    event_top1 = np.argmax(prob_one_event)
                    event_top15 = prob_sort[-15:]
                    event_top30 = prob_sort[-30:]
                    if target == event_top1:
                        self.correct_num_top1 += 1
                    if target in event_top15:
                        self.correct_num_top15 += 1
                    if target in event_top30:
                        self.correct_num_top30 += 1
                    self.mrr_sum += 1.0 / (np.where(prob_sort[::-1] == target)[0][0]+1)

            time_hamming, time_pow = predict_accuracy(delta_lambda, target_time, self.max_time, mask)
            self.total_time_hamming += time_hamming
            self.total_time_pow += time_pow
            self.total_predict_num += predict_num
        else:
            last_elem_idx = np.sum(mask, -1) - 1
            batch_size = mask.shape[0]
            prob_last = prob_next_event[np.arange(batch_size), last_elem_idx, :]
            target_last = target_event[np.arange(batch_size), last_elem_idx]
            predict_num = batch_size
            for prob_one_event, target in zip(prob_last, target_last):
                prob_sort = np.argsort(prob_one_event)
                event_top1 = np.argmax(prob_one_event)
                event_top15 = prob_sort[-15:]
                event_top30 = prob_sort[-30:]
                if target == event_top1:
                    self.correct_num_top1 += 1
                if target in event_top15:
                    self.correct_num_top15 += 1
                if target in event_top30:
                    self.correct_num_top30 += 1
                self.mrr_sum += 1.0 / (np.where(prob_sort[::-1] == target)[0][0]+1)
            delta_lambda_last = delta_lambda[np.arange(batch_size), last_elem_idx, :]  # [batch,21]
            cum_delta_lambda = np.cumsum(delta_lambda_last, -1)  # [batch,21]
            cum_index = np.arange(0.5, 21 + 0.5, 1)[None, :]
            predict_time = np.sum(cum_index * delta_lambda_last * np.exp(-cum_delta_lambda), -1)  # [batch]
            target_time_last = target_time[np.arange(batch_size), last_elem_idx]  # [batch]
            self.total_time_hamming += np.sum(np.abs(predict_time-target_time_last))
            self.total_time_pow += np.sum((predict_time-target_time_last)**2)
            self.total_predict_num +=predict_num

    def epoch_evaluate(self, output_file_path, pre_text=None, verbose=True):
        self.acc_top1 = self.correct_num_top1 / self.total_predict_num
        self.acc_top15 = self.correct_num_top15 / self.total_predict_num
        self.acc_top30 = self.correct_num_top30 / self.total_predict_num
        self.mrr=self.mrr_sum/self.total_predict_num
        self.time_mae = self.total_time_hamming / self.total_predict_num
        self.time_rmse = np.sqrt(self.total_time_pow / self.total_predict_num)
        if verbose is True:
            print("epoch acc@1:{:.3f},acc@15:{:.3f},acc@30:{:.3f},mrr:{:.3f},mae:{:.3f},rmse:{:.3f}".format(
                self.acc_top1, self.acc_top15, self.acc_top30, self.mrr, self.time_mae, self.time_rmse))
        with open(output_file_path, 'a') as f:
            if pre_text is not None:
                f.write(pre_text)
            f.write(
                '{},"epoch acc@1:{:.3f},acc@15:{:.3f},acc@30:{:.3f},mrr:{:.3f},mae:{:.3f},rmse:{:.3f}"\n'.format(
                    time.ctime(), self.acc_top1, self.acc_top15, self.acc_top30, self.mrr, self.time_mae,
                    self.time_rmse))

class evaluate_no_survival_analyses():
    def __init__(self,validate_last=False):
        self.max_time = 21
        self.total_time_hamming = 0.0
        self.total_time_pow = 0.0
        self.total_predict_num = 0.0
        self.correct_num_top1 = 0.0
        self.correct_num_top15 = 0.0
        self.correct_num_top30 = 0.0
        self.mrr_sum = 0.0
        self.acc_top1 = None
        self.acc_top15 = None
        self.acc_top30 = None
        self.time_mae = None
        self.time_rmse = None
        self.validate_last=validate_last

    def batch_evaluate(self, prob_next_time, prob_next_event, mask, target_event, target_time):
        if self.validate_last is False:
            for prob_one_seq, prob_time_one_seq, mask_one_seq, target_one_seq, target_time_one_seq \
                    in zip(prob_next_event, prob_next_time, mask, target_event, target_time):
                for prob_one_event, target,prob_one_time,target_one_time in \
                    zip(prob_one_seq[np.where(mask_one_seq == 1)],
                        target_one_seq[np.where(mask_one_seq == 1)],
                        prob_time_one_seq[np.where(mask_one_seq==1)],
                        target_time_one_seq[np.where(mask_one_seq==1)]):
                    prob_sort=np.argsort(prob_one_event)
                    event_top1=np.argmax(prob_one_event)
                    event_top15=prob_sort[-15:]
                    event_top30=prob_sort[-30:]
                    if target == event_top1:
                        self.correct_num_top1 +=1
                    if target in event_top15:
                        self.correct_num_top15 +=1
                    if target in event_top30:
                        self.correct_num_top30 +=1
                    self.mrr_sum += 1.0 / (np.where(prob_sort[::-1] == target)[0][0]+1)
                    predict_time=np.argmax(prob_one_time)
                    self.total_time_hamming += np.abs(predict_time-target_one_time)
                    self.total_time_pow += (predict_time-target_one_time)**2
                    self.total_predict_num +=1

        else:
            last_elem_idx = np.sum(mask, -1) - 1
            batch_size = mask.shape[0]
            prob_last = prob_next_event[np.arange(batch_size), last_elem_idx, :]
            target_last = target_event[np.arange(batch_size), last_elem_idx]
            predict_num = batch_size
            for prob_one_event, target in zip(prob_last, target_last):
                prob_sort = np.argsort(prob_one_event)
                event_top1 = np.argmax(prob_one_event)
                event_top15 = prob_sort[-15:]
                event_top30 = prob_sort[-30:]
                if target == event_top1:
                    self.correct_num_top1 += 1
                if target in event_top15:
                    self.correct_num_top15 += 1
                if target in event_top30:
                    self.correct_num_top30 += 1
                self.mrr_sum += 1.0 / (np.where(prob_sort[::-1] == target)[0][0]+1)

            target_time_last=target_time[np.arange(batch_size),last_elem_idx]
            predict_time_last=np.argmax(prob_next_time,-1)[np.arange(batch_size),last_elem_idx]
            self.total_time_hamming += np.sum(np.abs(predict_time_last-target_time_last))
            self.total_time_pow += np.sum((predict_time_last-target_time_last)**2)
            self.total_predict_num +=predict_num

    def epoch_evaluate(self, output_file_path, pre_text=None, verbose=True):
        self.acc_top1 = self.correct_num_top1 / self.total_predict_num
        self.acc_top15 = self.correct_num_top15 / self.total_predict_num
        self.acc_top30 = self.correct_num_top30 / self.total_predict_num
        self.mrr=self.mrr_sum/self.total_predict_num
        self.time_mae = self.total_time_hamming / self.total_predict_num
        self.time_rmse = np.sqrt(self.total_time_pow / self.total_predict_num)
        if verbose is True:
            print("epoch acc@1:{:.3f},acc@15:{:.3f},acc@30:{:.3f},mrr:{:.3f},mae:{:.3f},rmse:{:.3f}".format(
                self.acc_top1, self.acc_top15, self.acc_top30, self.mrr, self.time_mae, self.time_rmse))
        with open(output_file_path, 'a') as f:
            if pre_text is not None:
                f.write(pre_text)
            f.write(
                '{},"epoch acc@1:{:.3f},acc@15:{:.3f},acc@30:{:.3f},mrr:{:.3f},mae:{:.3f},rmse:{:.3f}"\n'.format(
                    time.ctime(), self.acc_top1, self.acc_top15, self.acc_top30, self.mrr, self.time_mae,
                    self.time_rmse))

class time_error_analyses():
    def __init__(self):
        ## x is true value, y is predict value, the value is count
        self.count = {}

    def batch_count(self, true_time, delta_lambda, mask):
        predict_time = compute_survival_time(delta_lambda, 21)
        for true_time_oneseq, predict_time_oneseq, mask_oneseq in zip(true_time, predict_time, mask):
            for true_value, predict_value in zip(true_time_oneseq[np.where(mask_oneseq == 1)],
                                                 predict_time_oneseq[np.where(mask_oneseq == 1)]):

                if true_value not in self.count:
                    self.count[true_value] = [predict_value]
                else:
                    self.count[true_value].append(predict_value)
