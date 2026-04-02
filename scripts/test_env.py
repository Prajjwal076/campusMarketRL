"""Local smoke test for the campus market environment."""

from __future__ import annotations

import numpy as np

from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv
from campus_market_env.utils.enums import ShopTypeEnum


def main() -> None:
    rng = np.random.default_rng(7)
    shop_types = [shop_type.value for shop_type in ShopTypeEnum]

    env = CampusMarketEnv(seed=7)
    observation = env.reset(seed=7)
    print("reset:", observation.model_dump())

    for step_index in range(10):
        action = CampusMarketAction(
            price_adjustment=float(rng.uniform(-0.3, 0.4)),
            marketing_spend=float(rng.uniform(0.0, 800.0)),
            restock_amount=int(rng.integers(0, 80)),
            product_focus=str(rng.choice(shop_types)),
        )
        observation, reward, done, info = env.step(action)
        print(f"step {step_index} observation:", observation.model_dump())
        print(f"step {step_index} reward:", reward)
        print(f"step {step_index} done:", done)
        print(f"step {step_index} info:", info)
        if done:
            break


if __name__ == "__main__":
    main()
