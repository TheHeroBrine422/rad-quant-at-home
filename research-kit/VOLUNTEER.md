# Lending your GPU for research: what it means

Thank you for considering this. Please read this page before you start; it's short.

**This is not the rad-quant-at-home job queue.** The queue ([README](../README.md)) needs no trust: it runs fixed job
types, and we check every result. Lending research access is different: **we get a shell (a command line) on a
container on your machine** and run our own experiments there. Only do it if you know and trust us.

## What we can do

- Log in to the container over Tailscale (an encrypted network) as an ordinary user called `research`, from our own
  machines only.
- Use the GPU(s) you lend, the CPU threads and RAM you allow, and the disk space of one Docker volume.
- Download public data and code from the internet (Hugging Face models, Python packages, GitHub), run experiments,
  and copy our results back to our machines.

## What we cannot do

- **Reach your home network.** Firewall rules on your machine (`firewall/install-firewall.sh`) block the container
  from every private address (your router, NAS, printers, phones, other computers, the host itself) and from cloud
  metadata addresses. Only the public internet is open. `./run.sh selftest` proves this on your machine.
- **Reach your other devices on Tailscale, or yours from ours.** The container is its own Tailscale device, tagged
  `donor`, in *our* Tailscale network. Our rules let it reach nothing; it doesn't touch any Tailscale you run on the
  host. You don't need a Tailscale account.
- **Get administrator (root) rights.** We have no root in the container, no extra kernel privileges, no Docker
  access, and we can't change the software image (it's read-only).
- **See your files.** The container sees only its own volume, not your home folder or your disks.
- **Accept connections from the internet.** No port is opened on your machine or your router.
- **Use more than you allow.** The CPU, RAM and process limits are hard limits enforced by your kernel (`donor.env`).

## What you should know (the honest part)

- **A container is not a perfect sandbox.** It shares your kernel. A serious kernel bug could let someone break out.
  GPU drivers are a large and complex part of the kernel, and the container talks to them directly. We have locked
  down everything else we can, but this is why it's only for people who trust us.
- We don't put secrets or private data on your machine: public data and code only.
- As the owner, you can see everything in the container (`docker exec`, the volume on your disk). That's fine with us.
- The GPU runs hot under load, like a long gaming session, and uses power (an R9700 draws ~300 W).
- Our jobs are written to be stopped at any moment. Stopping the container never costs you anything and never hurts
  us.

## Setup (about 15 minutes, once)

You need Linux with Docker. AMD: the normal amdgpu driver. NVIDIA: the driver (570 or newer) and the NVIDIA Container
Toolkit. We send you an auth key for Tailscale and help you through it.

```bash
sudo ./firewall/install-firewall.sh   # the network and its firewall; stays across reboots
cp donor.env.example donor.env        # set DONOR_VARIANT, TS_HOSTNAME, TS_AUTHKEY, the limits
./run.sh up                           # downloads our image (~4 GB AMD, ~10 GB NVIDIA) the first time
./run.sh selftest                     # send us the table it prints
```

Rather build the image yourself from this source? `./build.sh rocm` (or `cuda`) before `./run.sh up`.

## Day to day

| you want to | run |
|---|---|
| play a game / need the GPU back now | `./run.sh down` (stops it; GPU, RAM and CPU are free immediately) |
| give it back | `./run.sh start` (our work resumes by itself) |
| freeze it for a few minutes | `./run.sh pause` / `./run.sh unpause` (CPU drops to zero, but GPU memory and RAM stay used) |
| see what it's doing | `./run.sh status`, `./run.sh logs`, `./run.sh shell` |
| check the protections again | `./run.sh selftest` |
| change the limits | edit `donor.env`, then `./run.sh remove && ./run.sh up` (the data is kept) |

It restarts by itself after a reboot. If you'd rather it didn't: `docker update --restart no donor-research`.

## Stopping for good

```bash
./run.sh purge                                     # the container and its data volume
sudo ./firewall/uninstall-firewall.sh --remove-network
docker image rm ghcr.io/theherobrine422/rad-quant-at-home-research:0.3.0-rocm   # or -cuda
```

That's everything; nothing else was changed on your machine. Please tell us too, so we can remove the device from
our Tailscale network (it can't do anything once your container is gone, but it keeps things tidy).
You can also stop at any time without telling us first.
