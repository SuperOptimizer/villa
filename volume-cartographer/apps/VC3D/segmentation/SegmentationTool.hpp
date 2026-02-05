#pragma once

class SegmentationTool
{
public:
    virtual ~SegmentationTool() = default;

    virtual void cancel() noexcept = 0;
    virtual bool isActive() const noexcept = 0;
};

