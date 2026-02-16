#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <functional>
#include <vector>

class CSurfaceCollection;

class VectorOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    using Provider = std::function<void(VolumeViewerBase*, OverlayBuilder&)>;

    explicit VectorOverlayController(CSurfaceCollection* surfaces, QObject* parent = nullptr);

    void addProvider(Provider provider);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    void collectDirectionHints(VolumeViewerBase* viewer, OverlayBuilder& builder) const;
    void collectSurfaceNormals(VolumeViewerBase* viewer, OverlayBuilder& builder) const;

    CSurfaceCollection* _surfaces;
    std::vector<Provider> _providers;
};
