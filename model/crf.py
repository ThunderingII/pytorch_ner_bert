import torch
import torch.nn as nn

START_TAG = '<START>'
STOP_TAG = '<STOP>'


class CRF(nn.Module):
    def __init__(self, tag_size, device, batch_first):
        super().__init__()
        self.batch_first = batch_first

        self.tagset_size = tag_size
        t = torch.randn(self.tagset_size, self.tagset_size, device=device)
        self.transitions = nn.Parameter(t)

        self.start_tag = nn.Parameter(
            torch.randn(self.tagset_size, device=device))
        self.end_tag = nn.Parameter(
            torch.randn(self.tagset_size, device=device))

    def _score_sentence(self, feats, tags, mask_x, len_seq):
        """
        :param feats: [len_seq, batch_size, tag_size]
        :param tags: [batch_size, len_seq]
        :param mask_x:
        :param len_seq:
        :return:
        """

        bs = feats.size()[1]
        # Gives the score of a provided tag sequence
        score = self.start_tag[tags[:, 0]] + feats[0][
            torch.arange(bs), tags[:, 0]]

        for i in range(1, len(feats)):
            feat = feats[i]
            mask = mask_x[i]
            n_socre = score + self.transitions[tags[:, i - 1], tags[:, i]] + \
                      feat[torch.arange(bs), tags[:, i]]
            score = torch.where(mask, n_socre, score)

        score += self.start_tag[tags[:, len_seq - 1]]

        return score

    def _forward_alg(self, feats, mask_x):
        '''
        :param feats: [seq_len, batch_size, n_labels]
        :param len_seq: [batch_size]
        :return:
        '''

        # [1, n_labels]
        score = self.start_tag.unsqueeze(0)

        # Iterate through the sentence
        for i, feat in enumerate(feats):
            # shape: (batch_size, 1, num_tags)
            broadcast_score = score.unsqueeze(1)

            # shape: (batch_size, num_tags, 1)
            broadcast_emissions = feat.unsqueeze(2)

            # broadcast_score + broadcast_emissions
            # [[s1,s2,s3]]     [[e1],[e2],[e3]]
            # [s1+e1+t11,s2+e1+t12,s3+e1+t13]
            # [s1+e2,s2+e2,s3+e2+t23]
            # [s1+e3,s2+e3,s3+e3+t33]

            next_score = broadcast_score + self.transitions + broadcast_emissions

            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=2)

            score = torch.where(mask_x[i].unsqueeze(1), next_score, score)

        score += self.transitions[self.tag_to_ix[STOP_TAG]]

        return torch.logsumexp(score, dim=1)

    def crf_log_loss(self, feats, tags, mask_x, len_w):
        if self.batch_first:
            feats = feats.transpose(0, 1)
            mask_x = mask_x.transpose(0, 1)
        forward_score = self._forward_alg(feats, mask_x)
        gold_score = self._score_sentence(feats, tags, mask_x, len_w)
        return forward_score - gold_score

    def forward(self, feats, mask_x, len_seq):
        if self.batch_first:
            feats = feats.transpose(0, 1)
            mask_x = mask_x.transpose(0, 1)
        return self._viterbi_decode(feats, mask_x, len_seq)

    def _viterbi_decode(self, feats, mask_x, len_seq):
        """
        :param feats: lstm result, [seq_len, batch_size, n_labels]
        :param mask_x:
        :param len_seq: sequence length
        :return: the highest score and the corresponding path
        """

        bs = feats.size()[1]
        # [batch_size,n_labels]
        score = torch.full((bs, self.tagset_size), -10000.,
                           device=feats.device)

        backpointers = torch.zeros_like(feats, device=feats.device,
                                        dtype=torch.int)
        # START_TAG has all of the score.
        score[:, self.tag_to_ix[START_TAG]] = 0.

        # Iterate through the sentence
        for i, feat in enumerate(feats):
            # shape: (batch_size, 1, num_tags)
            broadcast_score = score.unsqueeze(1)

            # shape: (batch_size, num_tags, 1)
            broadcast_emissions = feat.unsqueeze(2)

            # broadcast_score + broadcast_emissions
            # [[s1,s2,s3]]     [[e1],[e2],[e3]]
            # [s1+e1+t11,s2+e1+t12,s3+e1+t13]
            # [s1+e2,s2+e2,s3+e2+t23]
            # [s1+e3,s2+e3,s3+e3+t33]

            next_score = broadcast_score + self.transitions + broadcast_emissions
            # [batch_size, num_tags]
            best_score, backpointers[i] = torch.max(next_score, dim=2)

            score = torch.where(mask_x[i].unsqueeze(1), best_score, score)

        # [batch_size,n_labels]
        score += self.transitions[self.tag_to_ix[STOP_TAG]]

        path_score, last_tag_ids = torch.max(score, dim=-1)

        best_tags_list = []
        # change to [batch_size, len_seq]
        backpointers = backpointers.transpose(0, 1)

        for i_bs in range(bs):
            best_tags = [last_tag_ids[i_bs].item()]
            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for bp in reversed(backpointers[i_bs][1:len_seq[i_bs]]):
                best_last_tag = bp[best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        if path_score.is_cuda:
            path_score = path_score.cpu()

        return path_score.detach().numpy(), best_tags_list
