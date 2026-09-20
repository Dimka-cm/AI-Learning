#!/usr/bin/env python3
"""
Обучение глаз для интерфейса Unreal Engine 5.

Кадры собираются рендером из подлинных ассетов Epic (цвета StyleColors.cpp,
иконки Starship), разметка проставляется автоматически — мы сами рисуем
каждую панель и потому знаем её координаты точно.

    python3 pretrain_ui.py --samples 8000 --epochs 20

Проверить быстро:
    python3 pretrain_ui.py --samples 300 --epochs 3 --width 320 --height 180
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ue5 import icons
from ue5.model import LOCATE, UIVision
from ue5.render import MODE_ID, render


def collect(n, rng, h, w, verbose=True):
    """Собирает n кадров с разметкой."""
    X = np.zeros((n, 3, h, w), np.uint8)
    B = np.zeros((n, len(LOCATE) * 4), np.float32)
    M = np.zeros(n, np.int64)
    S = np.zeros(n, np.float32)
    I = np.zeros((n, len(icons.TOOLBAR)), np.float32)
    t0 = time.time()
    for i in range(n):
        f = render(w, h, rng)
        X[i] = f.img.transpose(2, 0, 1)
        for k, kind in enumerate(LOCATE):
            b = f.box(kind)
            if b is not None:
                B[i, k * 4:(k + 1) * 4] = b.norm(w, h)
        M[i] = MODE_ID[f.mode]
        S[i] = float(f.selected)
        for idx in f.icon_ids:
            I[i, idx] = 1.0
        if verbose and (i + 1) % max(1, n // 10) == 0:
            done = i + 1
            el = time.time() - t0
            print(f"  собрано {done}/{n}  ({el:.0f} с, "
                  f"осталось ~{el / done * (n - done):.0f} с)", flush=True)
    return X, B, M, S, I


def evaluate(model, X, B, M, S, I, batch=32):
    """Качество на новых кадрах: ошибка коробок, точность режима и иконок."""
    model.eval()
    dev = next(model.parameters()).device
    err = np.zeros(len(LOCATE), np.float64)
    mode_ok = sel_ok = icon_ok = 0
    n = len(X)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(X[i:i + batch]).float().div_(255.).to(dev)
            pb, pm, ps, pi = model(xb)
            gb = torch.from_numpy(B[i:i + batch]).to(dev)
            d = (pb - gb).abs().reshape(len(xb), len(LOCATE), 4).mean(dim=2)
            err += d.sum(dim=0).cpu().numpy()
            mode_ok += (pm.argmax(1).cpu().numpy() == M[i:i + batch]).sum()
            sel_ok += ((ps > 0).float().cpu().numpy() == S[i:i + batch]).sum()
            icon_ok += (((pi > 0).float().cpu().numpy())
                        == I[i:i + batch]).mean(axis=1).sum()
    model.train()
    return {"box": (err / n).tolist(), "mode": mode_ok / n,
            "sel": sel_ok / n, "icon": icon_ok / n}


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение глаз на интерфейсе UE5")
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--test-samples", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--cores", type=int, default=0)
    ap.add_argument("--ram-limit", type=int, default=0, help="МБ")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--out", default="ui_vision.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cores:
        torch.set_num_threads(args.cores)
    print(f"[cpu] потоков обучению: {torch.get_num_threads()}")

    if not icons.available():
        print("ОШИБКА: нет ассетов UE5. Сначала выполните:")
        print("  python3 tools/fetch_ue5_assets.py")
        return 1

    h, w = args.height, args.width
    n_test = args.test_samples or max(200, args.samples // 6)
    mb = (args.samples + n_test) * 3 * h * w / 1e6 + 1650
    print(f"Сбор: {args.samples} обучающих + {n_test} проверочных, {w}x{h}")
    print(f"Потребуется ОЗУ: ~{mb:.0f} МБ (кадры + torch + батч)")
    if args.ram_limit and mb > args.ram_limit:
        print(f"ОТКАЗ: больше лимита {args.ram_limit} МБ")
        return 1

    rng = np.random.default_rng(args.seed)
    X, B, M, S, I = collect(args.samples, rng, h, w)
    rng_t = np.random.default_rng(args.seed + 10_000)
    Xte, Bte, Mte, Ste, Ite = collect(n_test, rng_t, h, w, verbose=False)

    model = UIVision(h, w)
    print(f"параметров: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Bt = torch.from_numpy(B)
    Mt = torch.from_numpy(M)
    St = torch.from_numpy(S)
    It = torch.from_numpy(I)

    curve = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(X[idx]).float().div_(255.)
            pb, pm, ps, pi = model(xb)
            loss = (F.smooth_l1_loss(pb, Bt[idx])
                    + 0.5 * F.cross_entropy(pm, Mt[idx])
                    + 0.3 * F.binary_cross_entropy_with_logits(ps, St[idx])
                    + 0.3 * F.binary_cross_entropy_with_logits(pi, It[idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(idx)
        el = time.time() - t0
        line = (f"  эпоха {ep}/{args.epochs}  потеря {tot / len(X):.4f}  "
                f"({el:.0f} с, осталось ~{el / ep * (args.epochs - ep):.0f} с)")
        if args.eval_every and (ep % args.eval_every == 0
                                or ep == args.epochs):
            r = evaluate(model, Xte, Bte, Mte, Ste, Ite, args.batch)
            line += (f"  | режим {r['mode'] * 100:.0f}%  "
                     f"панели {np.mean(r['box']) * 100:.1f}%")
            curve.append({"epoch": ep, "loss": tot / len(X), **r})
        print(line, flush=True)

    rep = evaluate(model, Xte, Bte, Mte, Ste, Ite, args.batch)
    print("\n" + "=" * 66)
    print("ЧТО ГЛАЗА НАУЧИЛИСЬ ВИДЕТЬ В ИНТЕРФЕЙСЕ UE5")
    print("=" * 66)
    print(f"{'панель':16s} {'ошибка коробки':>16s} {'в пикселях':>14s}")
    print("-" * 66)
    for k, name in enumerate(LOCATE):
        e = rep["box"][k]
        print(f"{name:16s} {e:15.4f} {e * (w + h) / 2:13.1f}")
    print("-" * 66)
    print(f"{'режим редактора':16s} {rep['mode'] * 100:14.0f}%   "
          f"(наугад {100 / 4:.0f}%)")
    print(f"{'есть выделение':16s} {rep['sel'] * 100:14.0f}%   (наугад 50%)")
    print(f"{'иконки панели':16s} {rep['icon'] * 100:14.0f}%   "
          f"(доля верно узнанных)")
    print("=" * 66)

    torch.save({"conv": model.conv.state_dict(),
                "state": model.state_dict(),
                "width": w, "height": h,
                "locate": LOCATE, "icons": icons.TOOLBAR,
                "report": rep, "curve": curve}, args.out)
    json.dump({"report": rep, "curve": curve,
               "samples": args.samples, "epochs": args.epochs},
              open(args.out.replace(".pt", ".json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\nГлаза сохранены: {args.out}")
    print(f"Всего заняло: {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
