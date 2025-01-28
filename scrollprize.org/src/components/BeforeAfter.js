import React, { useState, useRef, useEffect } from "react";

const BeforeAfter = ({ beforeImage, afterImage }) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef(null);
  const isDragging = useRef(false);

  const handleMove = (event) => {
    if (!isDragging.current) return;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const position = (x / rect.width) * 100;
    setSliderPosition(Math.min(Math.max(position, 0), 100));
  };

  const handleTouchMove = (event) => {
    if (!isDragging.current) return;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const touch = event.touches[0];
    const x = touch.clientX - rect.left;
    const position = (x / rect.width) * 100;
    setSliderPosition(Math.min(Math.max(position, 0), 100));
  };

  useEffect(() => {
    const handleMouseUp = () => {
      isDragging.current = false;
    };

    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('touchend', handleMouseUp);

    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="h-80 rounded-xl relative inline-block overflow-hidden cursor-col-resize w-full max-w-4xl"
      onMouseMove={handleMove}
      onTouchMove={handleTouchMove}
      onMouseDown={() => (isDragging.current = true)}
      onTouchStart={() => (isDragging.current = true)}
      style={{
        userSelect: 'none',
      }}
    >
      <img
        src={afterImage}
        alt="After"
        className="absolute top-0 left-0 w-full h-full object-cover"
      />

      <img
        src={beforeImage}
        alt="Before"
        className="absolute top-0 left-0 w-full h-full object-cover"
        style={{
          clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`
        }}
      />

      <div
        className="absolute top-0 bottom-0 w-1 bg-orange-700"
        style={{ left: `${sliderPosition}%`, cursor: 'col-resize' }}
      >
        <div className="absolute top-1/2 -translate-y-1/2 left-1/2 -translate-x-1/2 w-6 h-6 bg-black rounded-full flex items-center justify-center">
          <div className="flex items-center gap-1">
            <div className="w-1 h-4 bg-orange-700"></div>
            <div className="w-1 h-4 bg-orange-700"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BeforeAfter;
