#pragma once

#include <QObject>

#include "vc/core/PointCollections.hpp"

class VCCollection : public QObject, public PointCollections
{
    Q_OBJECT

public:
    explicit VCCollection(QObject* parent = nullptr);
    ~VCCollection() override;

signals:
    void collectionChanged(uint64_t collectionId);
    void collectionsAdded(const std::vector<uint64_t>& collectionIds);
    void collectionRemoved(uint64_t collectionId);

    void pointAdded(const ColPoint& point);
    void pointChanged(const ColPoint& point);
    void pointRemoved(uint64_t pointId);

protected:
    void onCollectionsAdded(const std::vector<uint64_t>& ids) override;
    void onCollectionRemoved(uint64_t id) override;
    void onCollectionChanged(uint64_t id) override;
    void onPointAdded(const ColPoint& p) override;
    void onPointChanged(const ColPoint& p) override;
    void onPointRemoved(uint64_t id) override;
};
