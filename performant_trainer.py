from transformers import Seq2SeqTrainer


# This trainer significantly reduces the memory footprint during training by reducing the output tensor to the
# argmax'ed token only.
class PerformantTrainer(Seq2SeqTrainer):
    def __init__(self, keyword_count=1, *args, **kwargs):
        self.keyword_count = keyword_count
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss, ignore_keys=None):
        loss, logits, labels = super(PerformantTrainer, self).prediction_step(model, inputs, prediction_loss,
                                                                              ignore_keys)
        logits = logits[0].argmax(-1)
        return loss, logits, labels
