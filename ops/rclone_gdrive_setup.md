# rclone + Google Drive setup (one-time)

This is the simplest durable storage option if your VM is ephemeral.

## 1) Install rclone

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y rclone
```

If you cannot use `apt`, see the official install instructions:

```bash
curl -fsSL https://rclone.org/install.sh | sudo bash
```

## 2) Configure a Google Drive remote named `gdrive`

Run:

```bash
rclone config
```

Then:

- `n` (new remote)
- name: `gdrive`
- storage: `drive`
- follow the prompts for OAuth login

Verify:

```bash
rclone lsd gdrive:
```

## 3) Create a destination folder (optional)

```bash
rclone mkdir gdrive:world-modality-vla
```

## 4) Test a copy

```bash
mkdir -p /tmp/rclone_test && echo hello > /tmp/rclone_test/hello.txt
rclone copy /tmp/rclone_test gdrive:world-modality-vla/rclone_test
rclone ls gdrive:world-modality-vla/rclone_test
```

