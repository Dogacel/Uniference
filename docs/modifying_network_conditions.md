# Modifying Network Conditions

While running some on-device experiments, you might want to emulate a slower network condition. This section describes how you can do that.

## Use Linux `tc`

If your Linux installation supports using `tc`, you can use it to emulate a slower network setting on device.

```shell
# Replace eth0 with your device's socket name. I.e. this is enP8p1s0 for Jetson.
sudo tc qdisc add dev eth0 root handle 1: htb default 12
sudo tc class add dev eth0 parent 1: classid 1:12 htb rate 10mbit ceil 10mbit
```

For example Jetson Nano Orin's linux kernel doesn't contain the required qdisc kinds, therefore this setting doesn't work.

## Using OpenWRT

If your devices are connected to a router which has OpenWRT installed, you can control the network conditions directly from your router.

**TODO:** Verify this section once you get access to a OpenWRT router. 

1. Install `luci-app-sqm` or `sqm-scripts`.
2. In the SQM UI:
    1. Set Interface = wan.
    2. Set Download/ Upload rates slightly below your ISP nominal (e.g., if ISP upload is 10Mbps, set 9000 kbps).
    3. Select Queue discipline = cake and Link layer adaptation for your link (e.g., ethernet/pppoe).

Maybe `tc` also works on OpenWRT, need to double check.

## Using ToxiProxy (Doesn't work with PyTorch.distributed)

**Important:** Toxiproxy doesn't work with GLOO backend because it only uses 25001 for randezvous. Currently [there is no way to set communication ports for gloo backend](https://github.com/pytorch/gloo/issues/54). This section is kept for documenting our attempts.

```shell
wget https://github.com/Shopify/toxiproxy/releases/download/v2.7.0/toxiproxy-server-linux-arm64 -O toxiproxy-server
wget https://github.com/Shopify/toxiproxy/releases/download/v2.7.0/toxiproxy-cli-linux-arm64 -O toxiproxy-cli
chmod +x toxiproxy-server toxiproxy-cli

# On a separate session
./toxiproxy-server

./toxiproxy-cli create -l 0.0.0.0:5000 -u 192.168.1.14:25001 mylink

./toxiproxy-cli toxic add -t latency -a latency=10 -u mylink
./toxiproxy-cli toxic add -t latency -a latency=10 -d mylink
./toxiproxy-cli toxic add -t bandwidth -a rate=12500 -u mylink
./toxiproxy-cli toxic add -t bandwidth -a rate=12500 -d mylink

# On server
iperf3 -s -p 25001

# On client
iperf3 -c localhost -p 5000 -t 30

sudo apt install hping3
sudo hping3 -S -p 5000 -c 5 127.0.0.1
```