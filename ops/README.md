# Raspberry Pi Operations

## Connect to `pi5-dog`

The target device is a Raspberry Pi 5 with 8 GB of RAM. It runs Ubuntu Server
and is connected to the same Orbi LAN as the development computer.

## Role in `drobot2`

This Raspberry Pi is the primary onboard computer for `drobot2`. Its planned
responsibilities include:

- IMU integration and sensor-data handling
- Video capture, processing, and streaming
- CAN bus communication with the robot's embedded devices
- Coordination of the robot's onboard services

Workloads that need more GPU compute may be offloaded over the network to a
remote computer. The Pi remains the onboard integration point for the robot's
hardware and local services while the remote computer handles selected
compute-intensive processing.

Current connection details:

- Hostname: `pi5-dog`
- Hardware: Raspberry Pi 5, 8 GB RAM
- Current IP address: `192.168.1.56`
- SSH username: `rd`
- Wi-Fi SSID: `ORBI32`

From PowerShell on a computer connected to the same network:

```powershell
ssh rd@192.168.1.56
```

Enter the password configured in Raspberry Pi Imager. The SSH password prompt
does not display characters, dots, or asterisks while typing.

If mDNS/Avahi is working on the Pi, the hostname can be used instead:

```powershell
ssh rd@pi5-dog.local
```

The IP address is currently more reliable because hostname discovery has not
always worked on this network.

## Confirm network reachability

```powershell
ping 192.168.1.56
Test-NetConnection 192.168.1.56 -Port 22
```

Port 22 must report `TcpTestSucceeded: True` for SSH to work. Ping may be
blocked even when SSH is available.

If the address changes, open the Orbi app, find `pi5-dog`, and use its current
`192.168.1.x` address. Create an Orbi DHCP reservation for `192.168.1.56` to
keep the address stable.

## Enable hostname discovery

After connecting by IP, install and enable Avahi on the Pi:

```bash
sudo apt update
sudo apt install -y avahi-daemon
sudo systemctl enable --now avahi-daemon
```

The hostname connection should then be available as `pi5-dog.local`.

## Reimage and host-key recovery

Reimaging the card changes the Pi's SSH host key. If SSH reports that the host
identification changed, remove only the old entries and reconnect:

```powershell
ssh-keygen -R 192.168.1.56
ssh-keygen -R pi5-dog.local
ssh rd@192.168.1.56
```

Only remove a stored key when the Pi was intentionally reimaged or its host key
was otherwise knowingly replaced.

## Offline troubleshooting

If Orbi shows the Pi as offline or with address `0.0.0.0`:

1. Allow up to five minutes after first boot for Ubuntu cloud-init setup.
2. Check the Ethernet cable and link/activity lights when using wired access.
3. Confirm the Pi uses a Raspberry Pi 5-compatible 64-bit image and adequate
   USB-C power supply.
4. Inspect the green activity LED for a repeating boot-error pattern.
5. Check the SD card's boot and cloud-init configuration before reimaging.
