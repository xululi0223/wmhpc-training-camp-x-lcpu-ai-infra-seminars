"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""


def run(program):
    regs = [i for i in range(1, 32+1)]
    cycles = 0

    def run_inst(inst, mask=None):
        if mask is None:
            mask = [True] * 32
        inst_type = inst[0]
        if inst_type == "add":
            regs[mask] += inst[1]
            cycles += 1
        elif inst_type == "mul":
            regs[mask] *= inst[1]
            cycles += 1
        elif inst_type == "if_lt":
            mask = regs < inst[1]
            if all(mask):
                run_inst(inst[2], mask)
            elif not any(mask):
                continue
            else:
                run_inst(inst[2], mask)
                run_inst(inst[3], ~mask)

    for inst in program:
        run_inst(inst)

    return regs, cycles