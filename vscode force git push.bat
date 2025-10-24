@echo off
REM =============================================
REM Auto-create VS Code Force Push Extension
REM With option to push only top-level files
REM =============================================
SET EXT_PATH=%USERPROFILE%\.vscode\extensions\force-push-button

REM Create extension directory
if not exist "%EXT_PATH%" mkdir "%EXT_PATH%"

REM Create package.json for the extension
(
echo {
echo   "name": "force-push-button",
echo   "displayName": "Force Push Button",
echo   "version": "1.0.6",
echo   "publisher": "local",
echo   "engines": { "vscode": "^1.60.0" },
echo   "activationEvents": ["onStartupFinished"],
echo   "main": "./extension.js",
echo   "contributes": {
echo     "commands": [
echo       {
echo         "command": "forcePush.push",
echo         "title": "🔥 Force Push"
echo       },
echo       {
echo         "command": "forcePush.setupRepo",
echo         "title": "🔗 Connect Repo"
echo       },
echo       {
echo         "command": "forcePush.selectBranch",
echo         "title": "🌿 Select Branch"
echo       }
echo     ]
echo   }
echo }
) > "%EXT_PATH%\package.json"

REM Create extension.js
(
echo const vscode = require('vscode'^);
echo const { exec } = require('child_process'^);
echo const path = require('path'^);
echo const fs = require('fs'^);
echo.
echo function activate(context^) {
echo     // Connect Repo Button
echo     let connectRepoButton = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 102^);
echo     connectRepoButton.text = '🔗 Connect Repo';
echo     connectRepoButton.command = 'forcePush.setupRepo';
echo     connectRepoButton.tooltip = 'Connect to Git repository';
echo     connectRepoButton.show(^);
echo.
echo     // Select Branch Button
echo     let selectBranchButton = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 101^);
echo     selectBranchButton.text = '🌿 Select Branch';
echo     selectBranchButton.command = 'forcePush.selectBranch';
echo     selectBranchButton.tooltip = 'Select or create a branch';
echo     selectBranchButton.show(^);
echo.
echo     // Force Push Button
echo     let forcePushButton = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100^);
echo     forcePushButton.text = '🔥 Force Push';
echo     forcePushButton.command = 'forcePush.push';
echo     forcePushButton.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground'^);
echo     forcePushButton.tooltip = 'Force push all changes to remote';
echo     forcePushButton.show(^);
echo.
echo     // Connect Repo Command
echo     let setupRepoCommand = vscode.commands.registerCommand('forcePush.setupRepo', async (^) =^> {
echo         const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
echo         if (!workspaceFolder^) {
echo             vscode.window.showErrorMessage('❌ No folder open. Open a folder first.'^);
echo             return;
echo         }
echo.
echo         let repoUrl = await vscode.window.showInputBox({
echo             prompt: 'Enter your GitHub repository URL',
echo             placeHolder: 'https://github.com/username/repo or https://github.com/username/repo.git',
echo             validateInput: (value^) =^> {
echo                 if (!value^) return 'URL cannot be empty';
echo                 if (!value.includes('github.com'^) ^&^& !value.includes('gitlab.com'^) ^&^& !value.includes('bitbucket.org'^)^) {
echo                     return 'Please enter a valid Git repository URL';
echo                 }
echo                 return null;
echo             }
echo         }^);
echo.
echo         if (!repoUrl^) return;
echo.
echo         // Auto-append .git if not present
echo         repoUrl = repoUrl.trim(^).replace(/\/$/, ''^); // Remove trailing slash
echo         if (!repoUrl.endsWith('.git'^)^) {
echo             repoUrl += '.git';
echo         }
echo.
echo         const terminal = vscode.window.createTerminal({ name: 'Connect Repo', cwd: workspaceFolder }^);
echo         terminal.show(^);
echo.
echo         const gitDir = path.join(workspaceFolder, '.git'^);
echo         if (!fs.existsSync(gitDir^)^) {
echo             terminal.sendText('git init'^);
echo         }
echo.
echo         terminal.sendText('git remote remove origin 2^>nul ^|^| echo Removing old remote...'^);
echo         terminal.sendText(`git remote add origin ${repoUrl}`^);
echo.
echo         vscode.window.showInformationMessage(`✅ Connected to: ${repoUrl}. Now click "🌿 Select Branch".`^);
echo     }^);
echo.
echo     // Select Branch Command
echo     let selectBranchCommand = vscode.commands.registerCommand('forcePush.selectBranch', async (^) =^> {
echo         const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
echo         if (!workspaceFolder^) {
echo             vscode.window.showErrorMessage('❌ No folder open. Open a folder first.'^);
echo             return;
echo         }
echo.
echo         const gitDir = path.join(workspaceFolder, '.git'^);
echo         if (!fs.existsSync(gitDir^)^) {
echo             vscode.window.showErrorMessage('❌ Not a Git repository. Click "🔗 Connect Repo" first.'^);
echo             return;
echo         }
echo.
echo         // Get list of branches
echo         exec('git branch -a', { cwd: workspaceFolder }, async (err, stdout^) =^> {
echo             let branches = [];
echo             if (!err ^&^& stdout^) {
echo                 branches = stdout.split('\n'^)
echo                     .map(b =^> b.replace('*', ''^).trim(^)^)
echo                     .filter(b =^> b ^&^& !b.includes('HEAD'^)^)
echo                     .map(b =^> b.replace('remotes/origin/', ''^)^);
echo                 branches = [...new Set(branches^)]; // Remove duplicates
echo             }
echo.
echo             // Add option to create new branch
echo             const options = [
echo                 { label: '➕ Create New Branch', description: 'Enter a custom branch name' },
echo                 { label: 'main', description: 'Default branch' },
echo                 ...branches.filter(b =^> b !== 'main'^).map(b =^> ({ label: b }^)^)
echo             ];
echo.
echo             const selection = await vscode.window.showQuickPick(options, {
echo                 placeHolder: 'Select a branch or create a new one'
echo             }^);
echo.
echo             if (!selection^) return;
echo.
echo             let branchName;
echo             if (selection.label === '➕ Create New Branch'^) {
echo                 branchName = await vscode.window.showInputBox({
echo                     prompt: 'Enter new branch name',
echo                     placeHolder: 'feature-branch',
echo                     validateInput: (value^) =^> {
echo                         if (!value^) return 'Branch name cannot be empty';
echo                         if (value.includes(' '^)^) return 'Branch name cannot contain spaces';
echo                         return null;
echo                     }
echo                 }^);
echo                 if (!branchName^) return;
echo             } else {
echo                 branchName = selection.label;
echo             }
echo.
echo             const terminal = vscode.window.createTerminal({ name: 'Select Branch', cwd: workspaceFolder }^);
echo             terminal.show(^);
echo             terminal.sendText(`git checkout -b ${branchName} 2^>nul ^|^| git checkout ${branchName}`^);
echo.
echo             vscode.window.showInformationMessage(`✅ Switched to branch: ${branchName}`^);
echo         }^);
echo     }^);
echo.
echo     // Force Push Command
echo     let forcePushCommand = vscode.commands.registerCommand('forcePush.push', async (^) =^> {
echo         const workspaceFolder = vscode.workspace.workspaceFolders?.find(folder =^>
echo             vscode.window.activeTextEditor?.document.uri.fsPath.startsWith(folder.uri.fsPath^)
echo         ^)?.uri.fsPath ^|^| vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
echo.
echo         if (!workspaceFolder^) {
echo             vscode.window.showErrorMessage('❌ No folder open. Open the repo you want to push to first.'^);
echo             return;
echo         }
echo.
echo         const gitDir = path.join(workspaceFolder, '.git'^);
echo         if (!fs.existsSync(gitDir^)^) {
echo             vscode.window.showErrorMessage('❌ Not a Git repository. Click "🔗 Connect Repo" first.'^);
echo             return;
echo         }
echo.
echo         exec('git remote', { cwd: workspaceFolder }, (err, stdout^) =^> {
echo             if (err ^|^| !stdout.trim(^)^) {
echo                 vscode.window.showErrorMessage('❌ No remote configured. Click "🔗 Connect Repo" first.'^);
echo                 return;
echo             }
echo.
echo             exec('git branch --show-current', { cwd: workspaceFolder }, async (err, branch^) =^> {
echo                 if (err ^|^| !branch.trim(^)^) {
echo                     vscode.window.showErrorMessage('❌ No branch checked out. Click "🌿 Select Branch" first.'^);
echo                     return;
echo                 }
echo.
echo                 const currentBranch = branch.trim(^);
echo.
echo                 // Ask user what to push
echo                 const pushOption = await vscode.window.showQuickPick([
echo                     { label: '📁 Push Everything', description: 'All files and subdirectories' },
echo                     { label: '📄 Push Top-Level Only', description: 'Only files in root folder (no subdirectories^)' }
echo                 ], {
echo                     placeHolder: `Choose what to push to "${currentBranch}"`
echo                 }^);
echo.
echo                 if (!pushOption^) return;
echo.
echo                 const pushTopLevelOnly = pushOption.label === '📄 Push Top-Level Only';
echo.
echo                 vscode.window.showWarningMessage(
echo                     pushTopLevelOnly 
echo                         ? `⚠️ Force push TOP-LEVEL files only to "${currentBranch}"?`
echo                         : `⚠️ Force push ALL changes to "${currentBranch}"?`,
echo                     'Yes, Force Push',
echo                     'Cancel'
echo                 ^).then(selection =^> {
echo                     if (selection === 'Yes, Force Push'^) {
echo                         const terminal = vscode.window.createTerminal({ name: 'Force Push', cwd: workspaceFolder }^);
echo                         terminal.show(^);
echo.
echo                         if (pushTopLevelOnly^) {
echo                             // Add only top-level files
echo                             terminal.sendText('git add . --ignore-submodules'^);
echo                             terminal.sendText('git reset -- */* 2^>nul ^|^| echo Excluding subdirectories...'^);
echo                             terminal.sendText(`git commit -m "Force overwrite on ${currentBranch} (top-level only^)" --allow-empty`^);
echo                         } else {
echo                             // Add everything
echo                             terminal.sendText('git add -A'^);
echo                             terminal.sendText(`git commit -m "Force overwrite on ${currentBranch}" --allow-empty`^);
echo                         }
echo.
echo                         terminal.sendText(`git push origin ${currentBranch} --force`^);
echo                         vscode.window.showInformationMessage(`🔥 Force pushing to "${currentBranch}"...`^);
echo                     }
echo                 }^);
echo             }^);
echo         }^);
echo     }^);
echo.
echo     context.subscriptions.push(forcePushCommand, setupRepoCommand, selectBranchCommand, forcePushButton, connectRepoButton, selectBranchButton^);
echo }
echo.
echo function deactivate(^) {}
echo.
echo module.exports = { activate, deactivate };
) > "%EXT_PATH%\extension.js"

echo.
echo ✅ Extension created at: %EXT_PATH%
echo.
echo ========================================
echo NEXT STEPS:
echo 1. RESTART VS CODE
echo 2. Click "🔗 Connect Repo"
echo 3. Click "🌿 Select Branch"
echo 4. Click "🔥 Force Push"
echo 5. Choose: "📁 Push Everything" or "📄 Push Top-Level Only"
echo ========================================
echo.
pause