import unittest
import torch
import os
import tempfile
import shutil
import numpy as np
from typing import Dict

from pyhealth.models import ResNetLSTM, CNNLSTM
from pyhealth.datasets import create_sample_dataset


class TestConv2dResNetLSTM(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

        # TUH-like fake structure 
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = os.path.join(
            self.temp_dir,
            "data/tuh_eeg/tuh_eeg_seizure/v2.0.5/edf/dev"
        )
        os.makedirs(self.base_path, exist_ok=True)

        subject = "aaaaaajy"
        session = "s001_2002"
        montage = "02_tcp_le"

        self.sample_dir = os.path.join(
            self.base_path, subject, session, montage
        )
        os.makedirs(self.sample_dir, exist_ok=True)

        # Fake EEG data 
        self.batch_size = 2
        self.seq_len = 120
        self.in_channel = 8
        self.output_dim = 3
        self.device = "cpu"

        signals = torch.tensor(np.random.randn(self.batch_size, self.seq_len, self.in_channel).astype(np.float32))
        print(signals.size())

        self.samples = []
        for i in range(self.batch_size):
            self.samples.append({
                "patient_id": f"p{i}",
                "signal": signals[i].tolist(),
                "label": torch.tensor([
                    'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG F7','EEG F8',
                    'EEG C3','EEG C4','EEG CZ','EEG T3','EEG T4',
                    'EEG P3','EEG P4','EEG O1','EEG O2','EEG T5','EEG T6','EEG PZ','EEG FZ'
                ]),
                # "label_bitgt_1": label_bitgt_1,
                # "label_bitgt_2": label_bitgt_2,
                # "label_name": label_name
            })

            task_name: str = "tusz_task"
            input_schema: Dict[str, str] = { "signal": "tensor" }
            output_schema: Dict[str, str] = {
                "label": "tensor",
                # "label_bitgt_1": "tensor",
                # "label_bitgt_2": "tensor",
                # "label_name": "text",
            }

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="tusz",
            task_name=task_name
        )

        self.model = CNNLSTM(
            dataset=self.dataset,
            encoder=None,
            num_layers=1,
            in_channel=self.in_channel,
            output_dim=self.output_dim,
            batch_size=self.batch_size,
            device=self.device,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    # BASIC FUNCTIONAL TESTS

    def test_forward(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        with torch.no_grad():
            output, hidden = self.model(x)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        self.assertEqual(len(hidden), 2)

    def test_hidden_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        _, (h, c) = self.model(x)

        self.assertEqual(h.shape, (1, self.batch_size, 256))
        self.assertEqual(c.shape, (1, self.batch_size, 256))

    # GRADIENT TEST

    def test_backward(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        output, _ = self.model(x)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        loss.backward()

        has_grad = any(
            p.grad is not None for p in self.model.parameters() if p.requires_grad
        )

        self.assertTrue(has_grad)

    # MINI TRAINING STEP

    def test_training_step(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()

        output1, _ = self.model(x)
        loss1 = loss_fn(output1, target)

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        output2, _ = self.model(x)
        loss2 = loss_fn(output2, target)

        # Loss should change after update (not necessarily decrease, but different)
        self.assertNotEqual(loss1.item(), loss2.item())

    # NUMERICAL STABILITY

    def test_output_is_finite(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        output, _ = self.model(x)

        self.assertTrue(torch.isfinite(output).all())

    # CONSISTENCY TEST

    def test_deterministic_forward(self):
        x = torch.randn(self.batch_size, self.seq_len, self.in_channel)

        torch.manual_seed(0)
        out1, _ = self.model(x)

        torch.manual_seed(0)
        out2, _ = self.model(x)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()