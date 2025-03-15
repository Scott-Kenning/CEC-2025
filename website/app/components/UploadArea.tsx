"use client";
import React, { useState, useEffect, ChangeEvent, DragEvent } from "react";

interface UploadAreaProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File, preview: string) => void;
  onRemoveFile: () => void;
}

export default function UploadArea({
  selectedFile,
  previewUrl,
  onFileSelect,
  onRemoveFile,
}: UploadAreaProps) {
  const [isDragging, setIsDragging] = useState<boolean>(false);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const preview = URL.createObjectURL(file);
    onFileSelect(file, preview);
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const preview = URL.createObjectURL(file);
      onFileSelect(file, preview);
    }
  };

  return (
    <>
      {!selectedFile ? (
        <div
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative flex flex-col items-center justify-center border-2 border-dashed rounded p-6 transition-colors h-64 ${
            isDragging ? "bg-gray-100 border-gray-400" : "border-gray-300"
          }`}
        >
          <p className="text-gray-600">
            Drag &amp; drop an image, or click to select a file
          </p>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
        </div>
      ) : (
        <div className="relative">
          <img
            src={previewUrl!}
            alt="Preview"
            className="max-h-64 object-contain border border-gray-200 rounded mx-auto"
          />
          <button
            type="button"
            onClick={onRemoveFile}
            className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full w-8 h-8 flex items-center justify-center"
          >
            &times;
          </button>
        </div>
      )}
    </>
  );
}
