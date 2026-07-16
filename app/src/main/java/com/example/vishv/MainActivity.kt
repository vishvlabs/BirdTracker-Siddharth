package com.example.vishv

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.vishv.ui.MainScreen
import com.example.vishv.ui.theme.VishVTheme
import com.example.vishv.viewmodel.MainViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VishVTheme {
                val vm: MainViewModel = viewModel()
                MainScreen(viewModel = vm)
            }
        }
    }
}
