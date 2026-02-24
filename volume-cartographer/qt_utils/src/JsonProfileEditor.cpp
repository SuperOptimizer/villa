#include "qt_utils/JsonProfileEditor.hpp"
#include "qt_utils/UserProfileStore.hpp"

#include <QComboBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QVBoxLayout>

namespace qt_utils
{

JsonProfileEditor::JsonProfileEditor(
    const QString& title,
    const QString& settingsGroup,
    QWidget* parent)
    : QGroupBox(title, parent)
    , settingsGroup_(settingsGroup)
{
    auto* layout = new QVBoxLayout(this);

    // Optional description label
    description_ = new QLabel(this);
    description_->setWordWrap(true);
    description_->setVisible(false);
    layout->addWidget(description_);

    // Profile selector row
    auto* profileRow = new QHBoxLayout();
    auto* profileLabel = new QLabel(tr("Profile:"), this);
    profileCombo_ = new QComboBox(this);
    profileCombo_->setToolTip(
        tr("Select a profile.\n"
           "- Custom: editable free-form text\n"
           "- Built-in profiles: read-only\n"
           "- User profiles: editable, saved across sessions"));
    profileRow->addWidget(profileLabel);
    profileRow->addWidget(profileCombo_, 1);
    layout->addLayout(profileRow);

    // User profile management buttons
    auto* buttonRow = new QHBoxLayout();
    saveAsBtn_ = new QPushButton(tr("Save As..."), this);
    renameBtn_ = new QPushButton(tr("Rename..."), this);
    deleteBtn_ = new QPushButton(tr("Delete"), this);
    buttonRow->addWidget(saveAsBtn_);
    buttonRow->addWidget(renameBtn_);
    buttonRow->addWidget(deleteBtn_);
    buttonRow->addStretch();
    layout->addLayout(buttonRow);

    connect(saveAsBtn_, &QPushButton::clicked, this, &JsonProfileEditor::onSaveAs);
    connect(renameBtn_, &QPushButton::clicked, this, &JsonProfileEditor::onRename);
    connect(deleteBtn_, &QPushButton::clicked, this, &JsonProfileEditor::onDelete);

    // JSON text editor
    textEdit_ = new QPlainTextEdit(this);
    textEdit_->setTabChangesFocus(true);
    layout->addWidget(textEdit_);

    // Validation status label
    statusLabel_ = new QLabel(this);
    statusLabel_->setWordWrap(true);
    statusLabel_->setVisible(false);
    statusLabel_->setStyleSheet(QStringLiteral("color: #c0392b;"));
    layout->addWidget(statusLabel_);

    // Connections
    connect(textEdit_, &QPlainTextEdit::textChanged, this, [this]() {
        handleTextEdited();
    });
    connect(
        profileCombo_,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        this,
        [this](int idx) {
            if (!profileCombo_ || idx < 0) {
                return;
            }
            const QString profileId = profileCombo_->itemData(idx).toString();
            applyProfile(profileId, true);
        });

    updateUserProfileButtons();
}

void JsonProfileEditor::setDescription(const QString& text)
{
    if (!description_) {
        return;
    }
    description_->setText(text);
    description_->setVisible(!text.trimmed().isEmpty());
}

void JsonProfileEditor::setPlaceholderText(const QString& text)
{
    if (textEdit_) {
        textEdit_->setPlaceholderText(text);
    }
}

void JsonProfileEditor::setTextToolTip(const QString& text)
{
    if (textEdit_) {
        textEdit_->setToolTip(text);
    }
}

void JsonProfileEditor::setProfiles(
    const QVector<Profile>& profiles,
    const QString& defaultProfileId)
{
    profiles_ = profiles;

    // Ensure a "Custom" profile exists
    bool hasCustom = false;
    for (const auto& profile : profiles_) {
        if (profile.id == QStringLiteral("custom")) {
            hasCustom = true;
            break;
        }
    }
    if (!hasCustom) {
        Profile customProfile;
        customProfile.id = QStringLiteral("custom");
        customProfile.label = tr("Custom");
        customProfile.editable = true;
        profiles_.prepend(customProfile);
    }

    loadUserProfiles();
    rebuildCombo();

    // Select default profile
    int defaultIndex = -1;
    for (int i = 0; i < profiles_.size(); ++i) {
        if (!defaultProfileId.isEmpty() && profiles_[i].id == defaultProfileId) {
            defaultIndex = i;
            break;
        }
    }

    if (profileCombo_->count() > 0) {
        if (defaultIndex < 0) {
            defaultIndex = profileCombo_->findData(QStringLiteral("custom"));
        }
        const int idx = defaultIndex >= 0 ? defaultIndex : 0;
        setProfile(profileCombo_->itemData(idx).toString(), false);
    }
}

void JsonProfileEditor::setSettingsPath(const QString& path)
{
    settingsPath_ = path;
}

void JsonProfileEditor::setProfile(const QString& profileId, bool fromUi)
{
    if (!profileCombo_) {
        return;
    }

    int idx = profileCombo_->findData(profileId);
    if (idx < 0) {
        idx = profileCombo_->findData(QStringLiteral("custom"));
    }
    if (idx < 0 && profileCombo_->count() > 0) {
        idx = 0;
    }

    if (idx >= 0) {
        const QSignalBlocker blocker(profileCombo_);
        profileCombo_->setCurrentIndex(idx);
        applyProfile(profileCombo_->itemData(idx).toString(), fromUi);
    }
}

auto JsonProfileEditor::profile() const -> QString
{
    return profileId_;
}

auto JsonProfileEditor::customText() const -> QString
{
    return customText_;
}

void JsonProfileEditor::setCustomText(const QString& text)
{
    customText_ = text;
    if (profileId_ != QStringLiteral("custom")) {
        return;
    }
    if (textEdit_) {
        const QSignalBlocker blocker(textEdit_);
        textEdit_->setPlainText(customText_);
    }
    validateCurrentText();
}

auto JsonProfileEditor::currentText() const -> QString
{
    return textEdit_ ? textEdit_->toPlainText() : QString();
}

auto JsonProfileEditor::jsonObject(QString* error) const -> std::optional<QJsonObject>
{
    if (error) {
        error->clear();
    }

    const QString trimmed = currentText().trimmed();
    if (trimmed.isEmpty()) {
        return std::nullopt;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(trimmed.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        if (error) {
            *error = tr("JSON parse error (byte %1): %2")
                         .arg(static_cast<qulonglong>(parseError.offset))
                         .arg(parseError.errorString());
        }
        return std::nullopt;
    }

    if (!doc.isObject()) {
        if (error) {
            *error = tr("JSON must be an object.");
        }
        return std::nullopt;
    }

    return doc.object();
}

auto JsonProfileEditor::isValid() const -> bool
{
    return errorText_.isEmpty();
}

auto JsonProfileEditor::errorText() const -> QString
{
    return errorText_;
}

void JsonProfileEditor::loadUserProfiles()
{
    // Remove existing user profiles from the in-memory list
    for (int i = profiles_.size() - 1; i >= 0; --i) {
        if (UserProfileStore::isUserProfileId(profiles_[i].id)) {
            profiles_.removeAt(i);
        }
    }

    // Load from settings
    const auto userProfiles = UserProfileStore::loadAll(settingsGroup_, settingsPath_);

    // Insert after Custom (index 0), before built-in presets
    int insertPos = 1;
    for (const auto& up : userProfiles) {
        Profile p;
        p.id = UserProfileStore::profileIdFromStorageId(up.storageId);
        p.label = up.name;
        p.jsonText = up.data;
        p.editable = true;
        profiles_.insert(insertPos, p);
        ++insertPos;
    }
}

// --- Private implementation ---

void JsonProfileEditor::applyProfile(const QString& profileId, bool fromUi)
{
    const Profile* prof = findProfile(profileId);
    const QString normalizedId = prof ? prof->id : QStringLiteral("custom");

    if (profileId_ == normalizedId && !fromUi) {
        return;
    }

    profileId_ = normalizedId;

    const bool isCustom = (profileId_ == QStringLiteral("custom"));
    const bool isUser = UserProfileStore::isUserProfileId(profileId_);
    const bool editable = isCustom || isUser;
    const QString text = isCustom ? customText_ : (prof ? prof->jsonText : QString());

    if (textEdit_) {
        const QSignalBlocker blocker(textEdit_);
        textEdit_->setReadOnly(!editable);
        textEdit_->setPlainText(text);
    }

    validateCurrentText();
    updateUserProfileButtons();
    emit profileChanged(profileId_);
}

void JsonProfileEditor::handleTextEdited()
{
    if (updatingProgrammatically_) {
        return;
    }

    const bool isCustom = (profileId_ == QStringLiteral("custom"));
    const bool isUser = UserProfileStore::isUserProfileId(profileId_);

    if (!isCustom && !isUser) {
        return;
    }

    if (isCustom) {
        customText_ = currentText();
    } else {
        // Auto-save user profile text
        const int storageId = UserProfileStore::storageIdFromProfileId(profileId_);
        if (storageId > 0) {
            const Profile* p = findProfile(profileId_);
            const QString name = p ? p->label : QString();
            const QString text = currentText();
            UserProfileStore::update(settingsGroup_, storageId, name, text, settingsPath_);

            // Update in-memory profile
            for (auto& prof : profiles_) {
                if (prof.id == profileId_) {
                    prof.jsonText = text;
                    break;
                }
            }
        }
    }

    validateCurrentText();
    emit textChanged();
}

void JsonProfileEditor::validateCurrentText()
{
    QString error;
    jsonObject(&error);
    errorText_ = error;
    updateStatusLabel();
}

auto JsonProfileEditor::findProfile(const QString& id) const -> const Profile*
{
    for (const auto& profile : profiles_) {
        if (profile.id == id) {
            return &profile;
        }
    }
    return nullptr;
}

void JsonProfileEditor::updateStatusLabel()
{
    if (!statusLabel_) {
        return;
    }
    if (errorText_.isEmpty()) {
        statusLabel_->clear();
        statusLabel_->setVisible(false);
        return;
    }
    statusLabel_->setText(errorText_);
    statusLabel_->setVisible(true);
}

void JsonProfileEditor::onSaveAs()
{
    bool ok = false;
    const QString name = QInputDialog::getText(
        this, tr("Save Profile As"), tr("Profile name:"),
        QLineEdit::Normal, QString(), &ok);
    if (!ok || name.trimmed().isEmpty()) {
        return;
    }

    const QString jsonText = currentText();
    const int storageId =
        UserProfileStore::save(settingsGroup_, name.trimmed(), jsonText, settingsPath_);
    const QString profileId = UserProfileStore::profileIdFromStorageId(storageId);

    loadUserProfiles();
    rebuildCombo();
    setProfile(profileId, false);
    emit userProfilesChanged();
}

void JsonProfileEditor::onRename()
{
    if (!isUserProfile()) {
        return;
    }

    const Profile* current = findProfile(profileId_);
    if (!current) {
        return;
    }

    bool ok = false;
    const QString name = QInputDialog::getText(
        this, tr("Rename Profile"), tr("New name:"),
        QLineEdit::Normal, current->label, &ok);
    if (!ok || name.trimmed().isEmpty()) {
        return;
    }

    const int storageId = UserProfileStore::storageIdFromProfileId(profileId_);
    if (storageId <= 0) {
        return;
    }

    UserProfileStore::update(
        settingsGroup_, storageId, name.trimmed(), current->jsonText, settingsPath_);

    const QString selectedId = profileId_;
    loadUserProfiles();
    rebuildCombo();
    setProfile(selectedId, false);
    emit userProfilesChanged();
}

void JsonProfileEditor::onDelete()
{
    if (!isUserProfile()) {
        return;
    }

    const Profile* current = findProfile(profileId_);
    if (!current) {
        return;
    }

    const auto result = QMessageBox::question(
        this, tr("Delete Profile"),
        tr("Delete profile \"%1\"?").arg(current->label),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (result != QMessageBox::Yes) {
        return;
    }

    const int storageId = UserProfileStore::storageIdFromProfileId(profileId_);
    if (storageId <= 0) {
        return;
    }

    UserProfileStore::remove(settingsGroup_, storageId, settingsPath_);

    loadUserProfiles();
    rebuildCombo();
    setProfile(QStringLiteral("custom"), false);
    emit userProfilesChanged();
}

void JsonProfileEditor::updateUserProfileButtons()
{
    const bool isUser = isUserProfile();
    if (renameBtn_) {
        renameBtn_->setVisible(isUser);
    }
    if (deleteBtn_) {
        deleteBtn_->setVisible(isUser);
    }
}

auto JsonProfileEditor::isUserProfile() const -> bool
{
    return UserProfileStore::isUserProfileId(profileId_);
}

void JsonProfileEditor::rebuildCombo()
{
    const QSignalBlocker blocker(profileCombo_);
    profileCombo_->clear();
    for (const auto& profile : profiles_) {
        profileCombo_->addItem(profile.label, profile.id);
    }
}

}  // namespace qt_utils
