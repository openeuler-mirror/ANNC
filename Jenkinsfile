// ============================================================================
// ANNC-Nightly-Build Jenkins Pipeline
// ============================================================================
// 部署与配置踩坑记录（请运维同学在首次启用前逐项确认）：
//
// 1. Jenkinsfile 必须提交并推送到远程仓库
//    - 若 Jenkins 报 "Jenkinsfile not found"，请检查当前分支的远程仓库是否
//      已包含 Jenkinsfile。本地修改未提交/推送会导致 Jenkins 拉不到文件。
//
// 2. 构建目录与产物目录统一放到 /home/jenkins，避免占用 root 分区
//    - 本流水线使用 customWorkspace: /home/jenkins/workspace/ANNC-Nightly-Build
//    - 构建产物放在 workspace 下的 .annc-nightly/ 目录中（即
//      /home/jenkins/workspace/ANNC-Nightly-Build/.annc-nightly）
//    - 首次运行前必须在 Jenkins 节点上执行：
//        sudo mkdir -p /home/jenkins/workspace
//        sudo chown -R jenkins:jenkins /home/jenkins
//    - 若报 "AccessDeniedException: /home/jenkins"，说明目录未创建或权限不对。
//
// 3. Agent 必须预装 Python 依赖
//    - tensorflow==2.15.0
//    - pybind11==2.11.1
//    - nanobind>=2.4,<3（LLVM/MLIR 21.1.3 的 Python 绑定要求 nanobind 2.4+）
//    - 建议为 jenkins 用户配置国内 PyPI 镜像，否则 TensorFlow 等大包下载极慢：
//        sudo -u jenkins mkdir -p /var/lib/jenkins/.config/pip
//        sudo -u jenkins tee /var/lib/jenkins/.config/pip/pip.conf <<'EOF'
//        [global]
//        index-url = https://mirrors.aliyun.com/pypi/simple/
//        trusted-host = mirrors.aliyun.com
//        EOF
//      （清华镜像实测返回 403，建议优先使用阿里云镜像）
//    - 使用 jenkins 用户安装：
//        sudo -u jenkins pip3 install -r requirements.txt
//    - 若 Deps Check 阶段报 "missing required python packages"，请按上述命令安装。
//
// 4. 磁盘空间要求
//    - 首次编译 LLVM 需要约 50GB 磁盘空间
//    - 请确保 /home 分区空间充足
//
// 5. 部分第三方仓库需要 Git 账号授权
//    - 当前配置方式：在 Jenkins 节点上为 jenkins 用户配置全局 git credential
//      helper，使 CMake FetchContent 拉取私有/受限第三方仓库（如 KDNN）时
//      能自动提供有权限的 Git 账号。
//    - 配置步骤（以 gitcode 为例）：
//        sudo cp /path/to/.git-credentials /var/lib/jenkins/.git-credentials
//        sudo chown jenkins:jenkins /var/lib/jenkins/.git-credentials
//        sudo chmod 600 /var/lib/jenkins/.git-credentials
//        sudo -u jenkins git config --global credential.helper store
//    - 验证凭据是否生效：
//        sudo -u jenkins git clone --depth 1 https://gitcode.com/boostkit/kdnn.git /tmp/kdnn-test
//    - 若 CMake configure 阶段报 "could not read Username for 'https://gitcode.com'"
//      或 401 / 403，请检查 jenkins 用户的 Git 凭据是否已正确配置。
//
// 6. 其他系统依赖
//    - agent 必须预装：cmake、clang、ninja-build
//    - 安装示例（openEuler）：
//        sudo dnf install -y cmake clang ninja-build
//    - 若 Prepare 阶段报 "clang: command not found" 或 Build 阶段报
//      "./build.sh: Permission denied"，请检查 clang 是否已安装、build.sh
//      是否具有可执行权限。
//    - 可参考 jenkins/README.md 中的落地说明
// ============================================================================

pipeline {
    // 将构建工作目录放到 /home 下，避免占用 root 分区空间。
    // 运行前请确保 /home/jenkins 目录存在且 jenkins 用户可读写。
    agent {
        node {
            label 'openeuler-aarch64'
            customWorkspace '/home/jenkins/workspace/ANNC-Nightly-Build'
        }
    }

    options {
        disableConcurrentBuilds()
        timeout(time: 6, unit: 'HOURS')
        buildDiscarder(logRotator(
            numToKeepStr: '10',
            artifactNumToKeepStr: '3'
        ))
    }

    triggers {
        cron('H 2 * * *')
    }

    parameters {
        booleanParam(
            name: 'CLEAN_BUILD',
            defaultValue: false,
            description: '是否清理 build 目录后全量重建'
        )
        booleanParam(
            name: 'RUN_TESTS',
            defaultValue: false,
            description: '是否运行 pytest/ctest（当前预留）'
        )
    }

    environment {
        // 构建产物放到 workspace 下的 .annc-nightly 目录中。
        // workspace 本身已配置在 /home/jenkins/workspace，因此仍然满足
        // "不占用 root 分区" 的要求；同时 Jenkins archiveArtifacts 只能归档
        // workspace 内的相对路径，放在这里才能被正常归档。
        ANNC_NIGHTLY_HOME = "${WORKSPACE}/.annc-nightly"
        INSTALL_PREFIX = "${env.ANNC_NIGHTLY_HOME}/install"
        BUILD_INFO_FILE = "${env.ANNC_NIGHTLY_HOME}/build_info.txt"
        BUILD_STATUS_FILE = "${env.ANNC_NIGHTLY_HOME}/build_status.json"
    }

    stages {
        stage('Prepare') {
            steps {
                script {
                    echo "Running on node: ${env.NODE_NAME}"
                }
                sh '''
                    set +x
                    mkdir -p "${ANNC_NIGHTLY_HOME}"

                    echo "===== Build Info =====" > "${BUILD_INFO_FILE}"
                    echo "BUILD_NUMBER: ${BUILD_NUMBER}" >> "${BUILD_INFO_FILE}"
                    echo "BUILD_URL: ${BUILD_URL}" >> "${BUILD_INFO_FILE}"
                    echo "GIT_BRANCH: ${BRANCH_NAME:-unknown}" >> "${BUILD_INFO_FILE}"
                    echo "WORKSPACE: ${WORKSPACE}" >> "${BUILD_INFO_FILE}"
                    echo "" >> "${BUILD_INFO_FILE}"
                    echo "===== Environment =====" >> "${BUILD_INFO_FILE}"
                    uname -a >> "${BUILD_INFO_FILE}"
                    cmake --version >> "${BUILD_INFO_FILE}" 2>&1 || true
                    ninja --version >> "${BUILD_INFO_FILE}" 2>&1 || true
                    clang --version >> "${BUILD_INFO_FILE}" 2>&1 || true
                    python3 --version >> "${BUILD_INFO_FILE}" 2>&1 || true
                    echo "=======================" >> "${BUILD_INFO_FILE}"
                    cat "${BUILD_INFO_FILE}"

                    # Clean previous install directory, keep build/ for incremental compile
                    rm -rf "${INSTALL_PREFIX}"
                '''
            }
        }

        stage('Deps Check') {
            steps {
                sh '''
                    set -e
                    python3 - <<'PY'
import sys
required = ['tensorflow', 'pybind11', 'nanobind']
missing = []
for mod in required:
    try:
        __import__(mod)
        print(f'[OK] {mod}')
    except ImportError as e:
        print(f'[MISSING] {mod}: {e}')
        missing.append(mod)

if missing:
    print(f'ERROR: missing required python packages: {missing}', file=sys.stderr)
    print('Please install them on the agent before running this pipeline.', file=sys.stderr)
    sys.exit(1)

print('All required python packages are available.')
PY
                '''
            }
        }

        stage('Build') {
            steps {
                script {
                    def cleanFlag = params.CLEAN_BUILD ? '--clean' : ''
                    sh """
                        set -e
                        ./build.sh \
                          --build-type Release \
                          --install-prefix "${env.INSTALL_PREFIX}" \
                          --no-install-deps \
                          ${cleanFlag}
                    """
                }
            }
        }

        stage('Install Verify') {
            steps {
                sh '''
                    set -e
                    echo "Checking install artifacts..."
                    test -f "${INSTALL_PREFIX}/bin/annc-opt"
                    test -f "${INSTALL_PREFIX}/bin/annc-asm"
                    test -f "${INSTALL_PREFIX}/bin/annc"
                    test -f "${INSTALL_PREFIX}/bin/annc-tf-pipeline"
                    test -f "${INSTALL_PREFIX}/bin/annc-verify"
                    test -f "${INSTALL_PREFIX}/bin/annc-converter"
                    echo "Install directory size:"
                    du -sh "${INSTALL_PREFIX}"
                '''
            }
        }

        stage('Tests') {
            when {
                expression { params.RUN_TESTS }
            }
            steps {
                sh '''
                    set -e
                    export PATH="${INSTALL_PREFIX}/bin:${PATH}"
                    echo "Running pytest..."
                    python3 -m pytest tests/ -v || true
                    echo "Tests stage completed (currently non-blocking)"
                '''
            }
        }

        stage('Archive') {
            steps {
                script {
                    def duration = currentBuild.duration / 1000 / 60
                    def installSize = sh(
                        script: "du -sm \"${env.INSTALL_PREFIX}\" | awk '{print \$1}'",
                        returnStdout: true
                    ).trim()

                    writeFile file: "${env.BUILD_STATUS_FILE}", text: """{
  "build_number": ${env.BUILD_NUMBER},
  "branch": "${env.BRANCH_NAME ?: 'unknown'}",
  "status": "${currentBuild.currentResult}",
  "duration_minutes": ${duration},
  "install_size_mb": ${installSize}
}"""
                }
                // install/ 目录约 100G+，归档到 /var/lib/jenkins 会占满 root 分区，
                // 因此只归档体积较小的构建信息文件。如需完整安装产物，请直接到
                // agent 的 /home/jenkins/workspace/ANNC-Nightly-Build/.annc-nightly/
                // 目录下获取。
                archiveArtifacts artifacts: '.annc-nightly/build_info.txt,.annc-nightly/build_status.json', fingerprint: true
            }
        }

        stage('Notify') {
            steps {
                echo 'Build succeeded. No notification sent on success.'
            }
        }
    }

    post {
        always {
            script {
                echo "Build completed with result: ${currentBuild.currentResult}"
            }
            sh '''
                # Keep build/ directory for next incremental build.
                # Clean up transient files if any.
                echo "Build completed" >> "${BUILD_INFO_FILE}" || true
            '''
        }

        failure {
            script {
                echo "Build failed: ${env.BUILD_URL}"

                // 邮件通知（需在 Jenkins 系统管理中配置 SMTP 服务器）
                // 插件 email-ext 已安装，但当前未配置 SMTP。
                // 配置完成后取消下面注释：
                // emailext(
                //     subject: "[ANNC Nightly Build FAILED] #${env.BUILD_NUMBER} - ${env.BRANCH_NAME}",
                //     body: """ANNC nightly build failed.
                //     Build: ${env.BUILD_URL}
                //     Branch: ${env.BRANCH_NAME}
                //     Node: ${env.NODE_NAME}
                //     Please check the console output for details.
                //     """,
                //     to: "${env.CHANGE_AUTHOR_EMAIL ?: 'team@example.com'}"
                // )

                // 钉钉/企业微信 webhook 通知示例
                // 需在 Jenkins Credentials 中创建 dingtalk-webhook-token 后再启用
                // withCredentials([string(credentialsId: 'dingtalk-webhook-token', variable: 'DING_TOKEN')]) {
                //     sh """
                //         curl -s -X POST \\
                //           "https://oapi.dingtalk.com/robot/send?access_token=\${DING_TOKEN}" \\
                //           -H 'Content-Type: application/json' \\
                //           -d '{
                //             "msgtype": "text",
                //             "text": {
                //               "content": "[ANNC Nightly Build FAILED] #\${BUILD_NUMBER} \${BUILD_URL}"
                //             }
                //           }'
                //     """
                // }
            }
        }

        unstable {
            echo "Build is unstable: ${env.BUILD_URL}"
        }
    }
}
