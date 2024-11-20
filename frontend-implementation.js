// api.js
const API_URL = 'http://localhost:5000';

export const generateImage = async (noiseParam1, noiseParam2) => {
    try {
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ noise_param1: noiseParam1, noise_param2: noiseParam2 }),
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        return data.image;
    } catch (error) {
        console.error('Error generating image:', error);
        throw error;
    }
};

export const transferStyle = async (imageFile, style) => {
    try {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('style', style);

        const response = await fetch(`${API_URL}/style-transfer`, {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        return data.image;
    } catch (error) {
        console.error('Error transferring style:', error);
        throw error;
    }
};

// app.js
import React, { useState } from 'react';
import { generateImage, transferStyle } from './api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, Paintbrush, ImageIcon, RefreshCw } from 'lucide-react';

const App = () => {
    const [generatedImage, setGeneratedImage] = useState(null);
    const [styledImage, setStyledImage] = useState(null);
    const [noiseParam1, setNoiseParam1] = useState(50);
    const [noiseParam2, setNoiseParam2] = useState(50);
    const [selectedStyle, setSelectedStyle] = useState('vangogh');
    const [loading, setLoading] = useState(false);

    const handleGenerate = async () => {
        try {
            setLoading(true);
            const imageData = await generateImage(noiseParam1, noiseParam2);
            setGeneratedImage(`data:image/png;base64,${imageData}`);
        } catch (error) {
            alert('Error generating image. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        try {
            setLoading(true);
            const imageData = await transferStyle(file, selectedStyle);
            setStyledImage(`data:image/png;base64,${imageData}`);
        } catch (error) {
            alert('Error transferring style. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-4">
            <Tabs defaultValue="synthesis" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="synthesis">Image Synthesis</TabsTrigger>
                    <TabsTrigger value="transfer">Style Transfer</TabsTrigger>
                </TabsList>

                <TabsContent value="synthesis">
                    <Card>
                        <CardHeader>
                            <CardTitle>Facial Image Synthesis</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-4">
                                    <div>
                                        <label className="text-sm font-medium">Noise Parameter 1</label>
                                        <Slider 
                                            value={[noiseParam1]} 
                                            onValueChange={([value]) => setNoiseParam1(value)}
                                            max={100} 
                                            step={1} 
                                            className="mt-2" 
                                        />
                                    </div>
                                    <div>
                                        <label className="text-sm font-medium">Noise Parameter 2</label>
                                        <Slider 
                                            value={[noiseParam2]} 
                                            onValueChange={([value]) => setNoiseParam2(value)}
                                            max={100} 
                                            step={1} 
                                            className="mt-2" 
                                        />
                                    </div>
                                    <Button 
                                        className="w-full" 
                                        onClick={handleGenerate}
                                        disabled={loading}
                                    >
                                        <RefreshCw className="mr-2 h-4 w-4" />
                                        Generate New Face
                                    </Button>
                                </div>
                                <div className="border-2 border-dashed rounded-lg p-4 flex items-center justify-center">
                                    {generatedImage ? (
                                        <img 
                                            src={generatedImage} 
                                            alt="Generated face" 
                                            className="max-w-full h-auto"
                                        />
                                    ) : (
                                        <div className="text-center">
                                            <ImageIcon className="mx-auto h-12 w-12 text-gray-400" />
                                            <p className="mt-2 text-sm text-gray-500">
                                                Generated image will appear here
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="transfer">
                    <Card>
                        <CardHeader>
                            <CardTitle>Artistic Style Transfer</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-4">
                                    <label className="block">
                                        <div className="border-2 border-dashed rounded-lg p-4 flex items-center justify-center cursor-pointer">
                                            <input
                                                type="file"
                                                className="hidden"
                                                accept="image/*"
                                                onChange={handleFileUpload}
                                                disabled={loading}
                                            />
                                            <div className="text-center">
                                                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                                                <p className="mt-2 text-sm text-gray-500">
                                                    Upload your image
                                                </p>
                                            </div>
                                        </div>
                                    </label>
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">Select Style</label>
                                        <select 
                                            className="w-full p-2 border rounded-md"
                                            value={selectedStyle}
                                            onChange={(e) => setSelectedStyle(e.target.value)}
                                            disabled={loading}
                                        >
                                            <option value="vangogh">Van Gogh</option>
                                            <option value="picasso">Picasso</option>
                                            <option value="monet">Monet</option>
                                        </select>
                                    </div>
                                </div>
                                <div className="border-2 border-dashed rounded-lg p-4 flex items-center justify-center">
                                    {styledImage ? (
                                        <img 
                                            src={styledImage} 
                                            alt="Styled image" 
                                            className="max-w-full h-auto"
                                        />
                                    ) : (
                                        <div className="text-center">
                                            <ImageIcon className="mx-auto h-12 w-12 text-gray-400" />
                                            <p className="mt-2 text-sm text-gray-500">
                                                Styled image will appear here
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
};

export default App;
