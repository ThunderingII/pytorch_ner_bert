import numpy as np

NONE = "O"


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tag_to_idx, idx_to_tag):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags_to_idx: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tag_to_idx[NONE]
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def evaluate(tag, tag_pred, tag_to_idx, idx_to_tag):
    """Evaluates performance on test set

    Args:
        test: dataset that yields tuple of (sentences, tags)

    Returns:
        metrics: (dict) metrics["acc"] = 98.4, ...

    """
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.

    accs += [a == b for (a, b) in zip(tag, tag_pred)]

    lab_chunks = set(get_chunks(tag, tag_to_idx, idx_to_tag))
    lab_pred_chunks = set(get_chunks(tag_pred, tag_to_idx, idx_to_tag))

    correct_preds += len(lab_chunks & lab_pred_chunks)
    total_preds += len(lab_pred_chunks)
    total_correct += len(lab_chunks)

    fpr = get_fpr(correct_preds, total_correct, total_preds)
    acc = np.mean(accs)
    fpr['acc'] = acc
    return fpr


def get_fpr(correct_preds, total_correct, total_preds):
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return {"f1": f1, 'presision': p, 'recall': r}
