#pragma once

#include "ViewerOverlayControllerBase.hpp"

class BBoxOverlayController final : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit BBoxOverlayController(QObject* parent = nullptr);

protected:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override;
};

