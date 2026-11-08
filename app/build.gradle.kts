plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.presenceguard.app"
    compileSdk = 34
    buildToolsVersion = "34.0.0"

    defaultConfig {
        applicationId = "com.presenceguard.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        ndk {
            // 只保留 arm64：x86_64 只对模拟器有意义，armeabi-v7a 是为 32 位老机型准备的。
            // GPU 委托又带了两个 ABI 的原生库，去掉 32 位可省约 35MB。
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    androidResources {
        // TFLite 需要通过 mmap 直接读取模型，必须保持未压缩
        noCompress += listOf("tflite", "lite")
    }

    packaging {
        resources {
            excludes += setOf(
                "META-INF/AL2.0",
                "META-INF/LGPL2.1",
                "META-INF/*.kotlin_module"
            )
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    val cameraxVersion = "1.4.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // 主引擎：ML Kit 姿态检测（bundled 版，模型内置，不依赖 Google Play 服务）
    implementation("com.google.mlkit:pose-detection-accurate:18.0.0-beta5")

    // 副引擎：TFLite 目标检测（COCO person 类）
    implementation("org.tensorflow:tensorflow-lite-task-vision:0.4.4")
    // GPU 委托：让 EfficientDet 跑在 Adreno 上。这个插件自带 libtensorflowlite_gpu_jni.so，
    // 不引入额外的 tensorflow-lite 运行时，因此不会和 task-vision 内静态链接的 runtime 冲突
    implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")
}
