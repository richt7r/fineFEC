import numpy as np
import utils
from pathlib import Path
import time
import threading
import codes.ldpc.decoders.python.python_decoders as ldpc_python_decoders
import formatters
import sys


def worker(config, data, error_counter, thread):
    if config["decoder"] == "spa":
        data = ldpc_python_decoders.spa(
            data,
            config["i1"],
            config["i2"],
            config["i3"],
            config["i4"],
            int(config["num_iterations"]),
        )
        error_counter[thread] += np.where(utils.llr_to_bit(data))[0].size
    # else:
    #     raise ValueError(
    #         f"{config['config_path']}: specified decoder not implemented"
    #     )


def simulus(config_path: Path):
    config = utils.build_config(config_path)

    if bool(config["dump_path"]) | bool(config["terminal_output"]):
        header = utils.build_header(config_path)
        if bool(config["dump_path"]):
            dump_file = open(config["dump_path"], "w")
            dump_file.close()
        if bool(config["terminal_output"]):
            print(header)

    np.random.seed(int(config["seed"]))
    i1, i2, i3, i4 = formatters.from_indices_to_sparce(Path(config["h_path"]))
    config["i1"], config["i2"], config["i3"], config["i4"] = i1, i2, i3, i4
    n = i2.max() + 1
    start, stop, step = np.array(config["snr_range"].split(":"), dtype=float)
    snrs = np.arange(start, stop, step)

    for snr in snrs:
        start_time = time.time()
        bits = 0
        errors = 0
        while errors < int(config["errors_to_point"]):
            jobs = []
            error_counter = np.zeros(int(config["num_threads"]), dtype=int)
            while jobs.__len__() < int(config["num_threads"]):
                data = np.zeros(n, dtype=int)
                llrs = utils.gen_llr_bpsk(data, snr)
                jobs.append(
                    threading.Thread(
                        target=worker,
                        args=(config, llrs, error_counter, jobs.__len__()),
                    )
                )
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            bits += n * int(config["num_threads"])
            errors += np.sum(error_counter)
            simspeed = bits / (time.time() - start_time)

            if bool(config["dump_path"]):
                dump_file = open(config["dump_path"], "w")
                dump_file.write(
                    header
                    + f"{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{simspeed/10**6:.5f} Mb/s\n"
                )
                dump_file.close()
            if bool(config["terminal_output"]):
                print(
                    f"{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{simspeed/10**6:.5f} Mb/s",
                    end="\r",
                )

        header += f"{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{simspeed/10**6:.5f} Mb/s\n"
        print(f"{snr:.2f}\t\t\t{(errors/bits):.5e}\t\t\t{simspeed/10**6:.5f} Mb/s")


if not sys.argv.__len__() - 1:
    raise ValueError(f"specify config path after {sys.argv[0]}")

config_path = Path(sys.argv[1])
simulus(config_path)
