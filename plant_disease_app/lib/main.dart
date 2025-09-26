import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:share_plus/share_plus.dart';
import 'dart:convert';

void main() {
  runApp(const PlantDiseaseApp());
}

class PlantDiseaseApp extends StatelessWidget {
  const PlantDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Plant Disease Detector',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _image;
  String _prediction = '';
  String _solution = '';
  final picker = ImagePicker();

  final url = Uri.parse("http://10.127.232.9:8000/predict/");

  // History list
  List<Map<String, dynamic>> history = [];

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _prediction = "Predicting... Please wait";
        _solution = "";
      });
      _uploadImage(_image!);
    }
  }

  Future<void> _uploadImage(File imageFile) async {
    try {
      var request = http.MultipartRequest('POST', url);
      request.files
          .add(await http.MultipartFile.fromPath('file', imageFile.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var data = jsonDecode(responseData);

        setState(() {
          _prediction = data["prediction"] ?? "Unknown";
          _solution = data["solution"] ?? "No solution available";

          // Add to history
          history.insert(0, {
            "image": imageFile,
            "prediction": _prediction,
            "solution": _solution,
            "time": DateFormat('hh:mm a').format(DateTime.now()),
          });
        });
      } else {
        setState(() {
          _prediction = "Error: Server returned ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _prediction = "Error: $e";
      });
    }
  }

  void _shareResult() {
    if (_prediction.isNotEmpty) {
      Share.share(
        "🌱 Plant Disease Detection Result:\n\n"
        "Prediction: $_prediction\n"
        "Solution: $_solution",
      );
    }
  }

  // 🔥 Reset history
  void _resetHistory() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Reset History?"),
        content: const Text("Are you sure you want to clear all history?"),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text("Cancel"),
          ),
          ElevatedButton(
            onPressed: () {
              setState(() {
                history.clear();
              });
              Navigator.of(ctx).pop();
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("History cleared")),
              );
            },
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            child: const Text("Clear"),
          ),
        ],
      ),
    );
  }

  Color _getCardColor(String prediction) {
    if (prediction.toLowerCase().contains("healthy")) {
      return Colors.green.shade100;
    } else if (prediction.toLowerCase().contains("error")) {
      return Colors.grey.shade300;
    } else {
      return Colors.red.shade100;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: Drawer(
        child: Column(
          children: [
            const DrawerHeader(
              decoration: BoxDecoration(color: Colors.green),
              child: Center(
                child: Text(
                  "📜 Prediction History",
                  style: TextStyle(fontSize: 20, color: Colors.white),
                ),
              ),
            ),
            Expanded(
              child: history.isEmpty
                  ? const Center(
                      child: Text("No history available"),
                    )
                  : ListView.builder(
                      itemCount: history.length,
                      itemBuilder: (context, index) {
                        final item = history[index];
                        return Card(
                          margin: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 4),
                          child: ListTile(
                            leading: Image.file(
                              item["image"],
                              width: 50,
                              height: 50,
                              fit: BoxFit.cover,
                            ),
                            title: Text(item["prediction"]),
                            subtitle: Text(
                              "${item["solution"]}\n⏰ ${item["time"]}",
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        );
                      },
                    ),
            ),
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: ElevatedButton.icon(
                onPressed: _resetHistory,
                icon: const Icon(Icons.delete_forever),
                label: const Text("Reset History"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  minimumSize: const Size(double.infinity, 50),
                ),
              ),
            ),
          ],
        ),
      ),
      appBar: AppBar(
        title: const Text("🌱 Plant Disease Detector"),
        centerTitle: true,
        actions: [
          IconButton(
            onPressed: _shareResult,
            icon: const Icon(Icons.share),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (_image != null)
              Card(
                color: _getCardColor(_prediction),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Image.file(_image!, height: 200),
                      const SizedBox(height: 16),
                      Text(
                        "Prediction: $_prediction",
                        style: const TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        "Solution: $_solution",
                        style: const TextStyle(fontSize: 16),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Camera"),
                  style: ElevatedButton.styleFrom(
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text("Gallery"),
                  style: ElevatedButton.styleFrom(
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
