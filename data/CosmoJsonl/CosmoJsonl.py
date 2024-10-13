import os
import datasets

class CosmoJsonl(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'text': datasets.Value('string')
            })
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(os.path.join(self.config.data_dir, 'Vienna_cosmo_train.jsonl'))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={'filepath': data_dir}
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            data = [json.loads(l) for l in f.readlines()]
            for idx, line in enumerate(data):
                yield idx, line
