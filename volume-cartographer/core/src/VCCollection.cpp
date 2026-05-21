#include "vc/ui/VCCollection.hpp"

VCCollection::VCCollection(QObject* parent)
    : QObject(parent)
{
}

VCCollection::~VCCollection() = default;

void VCCollection::onCollectionsAdded(const std::vector<uint64_t>& ids) { emit collectionsAdded(ids); }
void VCCollection::onCollectionRemoved(uint64_t id) { emit collectionRemoved(id); }
void VCCollection::onCollectionChanged(uint64_t id) { emit collectionChanged(id); }
void VCCollection::onPointAdded(const ColPoint& p) { emit pointAdded(p); }
void VCCollection::onPointChanged(const ColPoint& p) { emit pointChanged(p); }
void VCCollection::onPointRemoved(uint64_t id) { emit pointRemoved(id); }
