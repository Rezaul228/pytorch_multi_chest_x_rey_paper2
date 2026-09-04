# 🖥️ Remote Server Connection Setup Guide

## 📋 Server Information

- **Server Hostname:** `pascal.l3s.intra`
- **Username:** `abedin`
- **Project Path:** `/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2`
- **Conda Environment:** `multi_pytorch`
- **Job Scheduler:** SLURM (volta partition)

---

## 🚀 Step 1: SSH Connection Setup

### Option A: Basic SSH Connection (Password Required)

On your new laptop, connect to the server:

```bash
ssh abedin@pascal.l3s.intra
```

Or if you need to specify a different port or gateway:

```bash
ssh -p 22 abedin@pascal.l3s.intra
```

### Option B: SSH Key Setup (Passwordless Login) - **RECOMMENDED**

#### 1. Generate SSH Key (if you don't have one)

On your **new laptop**, run:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept default location (`~/.ssh/id_ed25519`), or specify a custom path.

#### 2. Copy Public Key to Server

**Option 2a: Using ssh-copy-id (easiest)**

```bash
ssh-copy-id abedin@pascal.l3s.intra
```

**Option 2b: Manual copy**

```bash
# On your laptop, display your public key
cat ~/.ssh/id_ed25519.pub

# Copy the output, then on the server:
ssh abedin@pascal.l3s.intra
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "PASTE_YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

#### 3. Test Passwordless Login

```bash
ssh abedin@pascal.l3s.intra
```

You should now connect without entering a password!

---

## 📁 Step 2: Access the Project

Once connected to the server:

```bash
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
```

---

## 🐍 Step 3: Setup Conda Environment

### Check if Conda is Available

```bash
which conda
```

### Activate the Environment

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
```

### Verify Environment

```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import config; print('Config loaded successfully')"
```

---

## 🔧 Step 4: Configure SSH Config (Optional but Recommended)

On your **new laptop**, create/edit `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Add the following configuration:

```
Host pascal
    HostName pascal.l3s.intra
    User abedin
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Now you can simply connect with:

```bash
ssh pascal
```

---

## 📦 Step 5: Sync Project Files (If Needed)

### Option A: Work Directly on Server (Recommended)

Just SSH into the server and work there. All files are already on the server.

### Option B: Sync Files to Local Machine

If you want to work locally and sync:

**Using rsync (one-way sync from server to laptop):**

```bash
# On your laptop
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
    abedin@pascal.l3s.intra:/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/ \
    ~/local_project_folder/
```

**Using git (if project is in git):**

```bash
# On server, check if it's a git repo
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
git remote -v

# On laptop, clone or pull
git clone <repository_url>
```

---

## 🧪 Step 6: Test the Setup

### Test 1: SSH Connection

```bash
ssh pascal  # or ssh abedin@pascal.l3s.intra
```

### Test 2: Access Project

```bash
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
ls -la
```

### Test 3: Activate Environment

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
python -c "import config; config.print_current_config()"
```

### Test 4: Check SLURM Access

```bash
squeue -u abedin
sinfo
```

### Test 5: Submit a Test Job

```bash
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
./submit_training_simple_v2.sh test_connection_check
```

---

## 🛠️ Step 7: VS Code Remote Development (Optional)

If you use VS Code, you can connect directly to the server:

### Install Remote-SSH Extension

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Remote - SSH"
4. Install it

### Connect to Server

1. Press `F1` or `Ctrl+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Enter: `abedin@pascal.l3s.intra` or `pascal` (if you configured SSH config)
4. VS Code will open a new window connected to the server
5. Open the project folder: `/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2`

### Benefits:
- Edit files directly on the server
- Run terminal commands in VS Code
- Use VS Code extensions on remote files
- Debug Python code remotely

---

## 📝 Step 8: Quick Reference Commands

### Connect to Server
```bash
ssh pascal  # or ssh abedin@pascal.l3s.intra
```

### Navigate to Project
```bash
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
```

### Activate Environment
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
```

### Submit Training Job
```bash
./submit_training_simple_v2.sh <experiment_name>
```

### Check Job Status
```bash
squeue -u abedin
```

### View Job Output
```bash
tail -f logs/training_<JOB_ID>.out
```

### Cancel Job
```bash
scancel <JOB_ID>
```

---

## 🔐 Security Notes

1. **SSH Keys:** Always use SSH keys instead of passwords when possible
2. **Key Passphrase:** Consider adding a passphrase to your SSH key for extra security
3. **File Permissions:** Keep `~/.ssh` directory permissions as `700` and `authorized_keys` as `600`
4. **VPN/Network:** You may need to be on the same network or VPN to access `pascal.l3s.intra`

---

## ❓ Troubleshooting

### Issue: "Connection refused" or "Host unreachable"
- **Solution:** Check if you're on the correct network/VPN
- Verify the hostname: `ping pascal.l3s.intra`

### Issue: "Permission denied (publickey)"
- **Solution:** Ensure your public key is in `~/.ssh/authorized_keys` on the server
- Check file permissions: `chmod 600 ~/.ssh/authorized_keys`

### Issue: "Command not found: conda"
- **Solution:** Source conda before activating: `source /opt/conda/etc/profile.d/conda.sh`

### Issue: "No module named 'config'"
- **Solution:** Make sure you're in the project directory: `cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2`

### Issue: "sbatch: command not found"
- **Solution:** SLURM may not be in PATH. Try: `module load slurm` or check with your system administrator

---

## 📞 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify all paths and hostnames
3. Ensure you have proper permissions
4. Contact your system administrator if network/access issues persist

---

**Last Updated:** 2025-01-21
**Server:** pascal.l3s.intra
**Project:** pytorch_multi_chest_x_rey_paper2

