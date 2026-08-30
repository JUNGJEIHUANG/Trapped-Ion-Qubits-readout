import argparse

import subprocess

import sys

from pathlib import Path


DEFAULT_PYTHON = sys.executable

DEFAULT_PRETRAIN = "Pre_train/Run_pretrain/best.pth"

DEFAULT_LABEL_DIR = "data/intersection_train_data/site_dia_labels"


def build_command(args, name, output_dir, extra_flags, from_scratch=False):

    cmd = [

        args.python,

        "train_main.py",

        "--model_arch",

        "site_dia",

        "--site_mask_weight",

        "0",

        "--site_dia_label_dir",

        args.site_dia_label_dir,

        "--sample_sizes",

        str(args.sample_size),

        "--epochs",

        str(args.epochs),

        "--batch_size",

        str(args.batch_size),

        "--output_dir",

        str(output_dir),

    ]

    if from_scratch:

        cmd.append("--from_scratch")

    else:

        cmd.extend(["--pretrained_ckpt", args.pretrained_ckpt, "--load_mode", args.load_mode])


    cmd.extend(extra_flags)

    if args.disable_amp:

        cmd.append("--disable_amp")

    return cmd


def main():

    args = parse_args()

    root = Path(args.output_root)

    root.mkdir(parents=True, exist_ok=True)


    # Matches Table VI of the Site-AIT paper exactly: one "full" row, then the
    # architectural block (6 rows), then the PSF-reconstruction-consistency
    # block (3 rows). Every row is an independently trained checkpoint.
    variants = [

        {

            "name": "full",

            "output": root / "full",

            "flags": [],

            "from_scratch": False,

        },

        # --- Architectural components (Table VI, first block) ---
        {

            "name": "global_cross_attn",

            "output": root / "global_cross_attn",

            "flags": ["--site_use_global_cross_attn"],

            "from_scratch": False,

        },

        {

            "name": "no_self_attn",

            "output": root / "no_self_attn",

            "flags": ["--dia_num_ion_attn_layers", "0"],

            "from_scratch": False,

        },

        {

            "name": "no_tok_offsets",

            "output": root / "no_tok_offsets",

            "flags": ["--site_disable_tok_offset"],

            "from_scratch": False,

        },

        {

            "name": "no_psf_mask",

            "output": root / "no_psf_mask",

            "flags": ["--site_disable_psf_mask"],

            "from_scratch": False,

        },

        {

            "name": "no_pretrain",

            "output": root / "no_pretrain",

            "flags": [],

            "from_scratch": True,

        },

        {

            "name": "no_coord_constraint",

            "output": root / "no_coord_constraint",

            "flags": ["--site_disable_coord_head"],

            "from_scratch": False,

        },

        # --- PSF reconstruction consistency loss (Table VI, second block) ---
        {

            "name": "no_recon_loss",

            "output": root / "no_recon_loss",

            "flags": ["--site_disable_recon_loss"],

            "from_scratch": False,

        },

        {

            "name": "recon_sigma_frozen",

            "output": root / "recon_sigma_frozen",

            "flags": ["--site_freeze_psf_sigma"],

            "from_scratch": False,

        },

        {

            "name": "recon_no_warmup",

            "output": root / "recon_no_warmup",

            "flags": ["--site_recon_no_warmup"],

            "from_scratch": False,

        },

    ]


    selected = set(v.strip() for v in args.only.split(",") if v.strip())

    if selected:

        variants = [v for v in variants if v["name"] in selected]

        unknown = selected - {v["name"] for v in variants}

        if unknown:

            raise ValueError(f"Unknown variant(s) in --only: {sorted(unknown)}")


    command_log = root / "retrained_ablation_commands.txt"

    with open(command_log, "w", encoding="utf-8") as f:

        for v in variants:

            cmd = build_command(

                args,

                name=v["name"],

                output_dir=v["output"],

                extra_flags=v["flags"],

                from_scratch=v["from_scratch"],

            )

            f.write(f"{v['name']}\n")

            f.write(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd) + "\n\n")


    print(f"Wrote command log: {command_log}")

    if args.dry_run:

        print("Dry run only. Commands were not executed.")

        return


    for v in variants:

        print("=" * 100)

        print(f"Training retrained ablation variant: {v['name']}")

        cmd = build_command(

            args,

            name=v["name"],

            output_dir=v["output"],

            extra_flags=v["flags"],

            from_scratch=v["from_scratch"],

        )

        print(" ".join(str(x) for x in cmd))

        subprocess.run(cmd, check=True)


    print("All selected retrained ablation variants finished.")

    print("Expected checkpoint paths:")

    for v in variants:

        print(f"  {v['name']}: {v['output'] / f'sample_{args.sample_size}' / 'best.pth'}")


def parse_args():

    parser = argparse.ArgumentParser(

        description="Train every formal Site-DIA ablation variant as a separate checkpoint."

    )

    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)

    parser.add_argument("--pretrained_ckpt", type=str, default=DEFAULT_PRETRAIN)

    parser.add_argument("--load_mode", type=str, default="backbone", choices=["full", "backbone"])

    parser.add_argument("--site_dia_label_dir", type=str, default=DEFAULT_LABEL_DIR)

    parser.add_argument("--output_root", type=str, default="outputs/retrained_ablation_nomask")

    parser.add_argument("--sample_size", type=int, default=50000)

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=48)

    parser.add_argument("--disable_amp", action="store_true")

    parser.add_argument(

        "--only",

        type=str,

        default="",

        help=(

            "Comma-separated subset: full,global_cross_attn,no_self_attn,no_tok_offsets,"

            "no_psf_mask,no_pretrain,no_coord_constraint,no_recon_loss,recon_sigma_frozen,"

            "recon_no_warmup"

        ),

    )

    parser.add_argument("--dry_run", action="store_true", help="Write commands but do not run training.")

    return parser.parse_args()


if __name__ == "__main__":

    main()

