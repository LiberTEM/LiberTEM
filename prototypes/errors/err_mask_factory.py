from libertem import api

if __name__ == "__main__":
    with api.Context() as ctx:
        ds = ctx.load(
            "blo",
            path="/home/clausen/Data/127.0.0.1.blo",
            tileshape=(1, 8, 144, 144),
        )
        analysis = ctx.create_mask_analysis(dataset=ds, factories=[lambda: 1/0])
        ctx.run(analysis)
