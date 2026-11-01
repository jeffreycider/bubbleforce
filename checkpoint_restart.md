Download a restart checkpoint from the [checkpoint bucket](https://huggingface.co/buckets/jacknzheng/bubbleforce), then set `training.resume_checkpoint` to its local directory.

For example, resume the `increased_momentum` experiment from checkpoint 2000 and measure at updates 2008, 2009, and 2010:

```bash
python run.py increased_momentum \
  training.resume_checkpoint=/path/to/captures/checkpoint_002000 \
  'measurement.updates=[2008,2009,2010]' \
  measurement.output_dir=runs/spectrum/increased_momentum_restart
```

Choose measurement updates after the checkpoint’s saved update. Results are saved to `measurement.output_dir`.

Configure the learning rate, schedule, and weight decay in `config/train_config.py`. We use Muon for attention and MLP weight matrices, and AdamW for token embeddings, output projection weights, biases, and normalization gains.
